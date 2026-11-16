#!/usr/bin/env python
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):  # один поток на процесс joblib
    os.environ.setdefault(_v, "1")
"""Читаемость сохранённых представлений пробой, обученной до сходимости, и с отбеливанием.

Для каждого сохранённого z (results/z/<набор>/<метод>_<lambda>_s<seed>.npy), а также для входа x:
  * логистическая регрессия отдельно на каждый отложенный концепт, решённая до сходимости методом Ньютона
    (logreg_newton.py: та же цель, что у sklearn LogisticRegression(C); max |градиент| / n < 1e-6);
    штраф L2 с силой 1/C, C выбирается для каждого концепта по AP на валидации из сетки 1e-5..1e1; признаки центрируются и
    делятся на общий скаляр (средняя дисперсия координаты = 1), что не меняет геометрию;
  * то же на отбеленных признаках: (x - mu) V diag((l + eps)^-1/2) V^T по обучающей части, eps = 1e-3 * среднее l;
  * метрики на тесте: AP по всем парам (micro), macro-AP (концепты с >= 5 положительными), AUC внутри класса задачи,
    AP по 200 меткам (единое C по 4-кратной кросс-валидации внутри этих 200), частотные ориентиры.
Результат: results/probe_<набор>.jsonl.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

import logreg_newton as N
import train as T
from collapse_check import within_class_auc

HERE = Path(__file__).resolve().parent
CS = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]  # общая сетка: отбеленным признакам нужен более сильный штраф
SETS = {  # набор z -> (данные, словарь отложенных концептов, эмбеддинг входа)
    "cebab_v2": ("cebab", "aspect", "mqa"), "goemotions_v2": ("goemotions", "emotion", "mqa"),
    "goemotions_humanAB_v2": ("goemotions", "emotionB", "mqa"),
    "cub_AB_v2": ("cub", "cubB", "inception"), "cub4_AB_v2": ("cub4", "cubB", "inception"),
    "cub4sem_AB_v2": ("cub4sem", "cubB", "inception"),
    "finetune_cub4_v2": ("cub4", "cubB", "resnet18"), "finetune_goemotions_v2": ("goemotions", "emotionB", "mqa"),
}


def _platt(s_cal, y_cal, s_apply):
    """Калибровка Платта: a * s + b по отложенным оценкам, чтобы оценки разных концептов были сравнимы (micro-AP)."""
    if y_cal.sum() == 0 or y_cal.sum() == len(y_cal):
        return s_apply
    cal = LogisticRegression(C=1e4, max_iter=1000).fit(s_cal[:, None], y_cal)
    return cal.decision_function(s_apply[:, None])


def _chunk(d1):
    return int(max(1, min(32, 4e7 / d1 ** 2)))


def _path(Xtr, Ytr, Xev_list, cols):
    """Путь по CS с тёплым стартом для столбцов cols; для каждого C -- оценки на каждой матрице из Xev_list."""
    W, out, conv = None, [], []
    for C in CS:
        W, ok = N.fit(Xtr, Ytr[:, cols], C, W0=W, chunk=_chunk(Xtr.shape[1]))
        conv.append(ok)
        out.append([Xe @ W for Xe in Xev_list])
    return out, conv


def micro_ap(S, Y):
    return T.concept_metrics(torch.as_tensor(S, dtype=torch.float32), torch.as_tensor(Y, dtype=torch.float32))["ap"]


def macro_ap(S, Y):
    return T.concept_metrics(torch.as_tensor(S, dtype=torch.float32), torch.as_tensor(Y, dtype=torch.float32))["macro_ap"]


def probe(X, Y, split, y_task, seed):
    """X, Y -- numpy (все примеры); возвращает метрики на тесте и оценки на тесте."""
    dev = T.DEV
    fit, val, te = (split[k].numpy() for k in ("fit", "val", "test"))
    Xt = {k: N.add_bias(torch.tensor(X[i], device=dev)) for k, i in (("fit", fit), ("val", val), ("test", te))}
    Yt = {k: torch.tensor(Y[i], device=dev) for k, i in (("fit", fit), ("val", val), ("test", te))}
    m = Y.shape[1]
    pos = Yt["fit"].sum(0)
    cols = ((pos > 0) & (pos < len(fit))).nonzero().squeeze(1)
    S = (Yt["fit"].mean(0, keepdim=True)).expand(len(te), m).clone()  # константа для вырожденных концептов
    Sv_best = torch.zeros(len(val), m, device=dev)
    best = torch.full((m,), -1.0, device=dev)
    bestC = torch.zeros(m, device=dev)
    scores, conv = _path(Xt["fit"], Yt["fit"], [Xt["val"], Xt["test"]], cols)
    for C, (Sv, Ste) in zip(CS, scores):
        apv = N.ap_columns(Sv, Yt["val"][:, cols]).nan_to_num(0.0)
        better = apv > best[cols]
        c = cols[better]
        best[c], bestC[c] = apv[better], C
        S[:, c], Sv_best[:, c] = Ste[:, better], Sv[:, better]
    S, Sv_best, Yv = S.cpu().numpy(), Sv_best.cpu().numpy(), Y[val]
    for j in cols.tolist():
        S[:, j] = _platt(Sv_best[:, j], Yv[:, j], S[:, j])
    met = T.concept_metrics(torch.tensor(S, dtype=torch.float32), torch.tensor(Y[te], dtype=torch.float32))
    out = {"C_median": float(bestC[cols].median()), "converged": float(np.mean(conv)), "ap": met["ap"],
           "macro_ap": met["macro_ap"],
           "auc_within": within_class_auc(torch.tensor(S), torch.tensor(Y[te]), y_task[torch.as_tensor(te)])}
    # 200 меток: единое C по 4-кратной кросс-валидации (macro-AP вне фолда), калибровка Платта по оценкам вне фолда
    g = torch.Generator().manual_seed(1000 + seed)
    few = fit[torch.randperm(len(fit), generator=g)[:200].numpy()]
    folds = np.arange(len(few)) % 4
    Xfew, Yfew = N.add_bias(torch.tensor(X[few], device=dev)), torch.tensor(Y[few], device=dev)
    oof = torch.zeros(len(CS), len(few), m, device=dev)
    conv200 = []
    for f in range(4):
        tr = torch.tensor(folds != f, device=dev)
        va_idx = torch.tensor(np.flatnonzero(folds == f), device=dev)
        p = Yfew[tr].sum(0)
        cf = ((p > 0) & (p < tr.sum())).nonzero().squeeze(1)
        sc, cv_ok = _path(Xfew[tr], Yfew[tr], [Xfew[va_idx]], cf)
        conv200 += cv_ok
        for k, (Sva,) in enumerate(sc):
            oof[k][va_idx[:, None], cf[None, :]] = Sva
    cv = [np.mean([macro_ap(oof[k][torch.tensor(folds == f, device=dev)].cpu(), Y[few][folds == f])
                   for f in range(4)]) for k in range(len(CS))]
    k = int(np.argmax(cv))
    p = Yfew.sum(0)
    c2 = ((p > 0) & (p < len(few))).nonzero().squeeze(1)
    S2 = (Yfew.mean(0, keepdim=True)).expand(len(te), m).clone()
    W, ok = N.fit(Xfew, Yfew[:, c2], CS[k], chunk=_chunk(Xfew.shape[1]))
    S2[:, c2] = Xt["test"] @ W
    S2, oofk = S2.cpu().numpy(), oof[k].cpu().numpy()
    for j in c2.tolist():
        S2[:, j] = _platt(oofk[:, j], Y[few, j], S2[:, j])
    m2 = T.concept_metrics(torch.tensor(S2, dtype=torch.float32), torch.tensor(Y[te], dtype=torch.float32))
    out.update(ap_200=m2["ap"], macro_ap_200=m2["macro_ap"], C_200=CS[k],
               converged_200=float(np.mean(conv200 + [ok])))
    del Xt, Yt, Xfew, Yfew, oof
    torch.cuda.empty_cache()
    return out, S


def whiten(X, fit):
    mu = X[fit].mean(0)
    Xc = X - mu
    cov = Xc[fit].t() @ Xc[fit] / len(fit)
    lam, V = torch.linalg.eigh(cov)
    lam = lam.clamp_min(0)
    eps = 1e-3 * lam.mean()
    return Xc @ (V * (lam + eps).rsqrt()) @ V.t()


def rescale(X, fit):
    """Центрирование и общий скаляр: средняя дисперсия координаты = 1 (геометрия не меняется)."""
    Xc = X - X[fit].mean(0)
    return Xc / Xc[fit].pow(2).mean().sqrt()


def spec(name):
    """Набор z -> (данные, словарь отложенных концептов, эмбеддинг входа)."""
    if name in SETS:
        return SETS[name]
    if name.startswith("goemotions_split_"):  # goemotions_split_r1 -> отложенные emotion_r1B
        return "goemotions", f"emotion_{name.rsplit('_', 1)[1]}B", "mqa"
    if name.startswith(("cebab_", "goemotions_")):  # доза-ответ, сетка lambda, ручки схлопывания, TF-IDF
        ds = name.split("_")[0]
        ev = {"cebab": "aspect", "goemotions": "emotion"}[ds]
        return ds, ("emotionB" if "humanAB" in name else ev), ("tfidf" if "tfidf" in name else "mqa")
    if name.startswith("finetune_sun"):  # SUN Attributes, обучаемый ResNet-18 layer4: finetune_sun4_v2 -> sun4
        return name.split("_")[1], "sunB", "resnet18"
    if name.startswith("finetune_cub"):  # обучаемый ResNet-18 layer4: finetune_cub4sem_v2 -> cub4sem
        return name.split("_")[1], "cubB", "resnet18"
    if name.startswith("finetune_reuters"):  # Reuters-21578: отложенные темы topicB
        return "reuters", "topicB", "mqa"
    if name.startswith("finetune_goemotions"):  # finetune_goemotions_length_r2_v2 -> отложенные emotion_r2B
        part = [p for p in name.split("_") if p[:1] in "rs" and p[1:2].isdigit() or p[:2] == "st" and p[2:].isdigit()]
        return "goemotions", (f"emotion_{part[0]}B" if part else "emotionB"), "mqa"
    if "_gsplit" in name:  # другие разбиения групп атрибутов CUB: cub4_gsplit1 -> cub_g1B
        return name.split("_")[0], f"cub_g{name.rsplit('gsplit', 1)[1]}B", "inception"
    if name.startswith(("cub4sem_", "cub4_", "cub_")):
        return name.split("_")[0], "cubB", "inception"
    raise KeyError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sets", nargs="*", default=list(SETS))
    ap.add_argument("--no-white", action="store_true", help="только без отбеливания")
    ap.add_argument("--white-input-only", action="store_true", help="с отбеливанием -- только вход")
    args = ap.parse_args()
    for name in args.sets:
        ds, ev, emb = spec(name)
        zdir = HERE / "results" / "z" / name
        if not zdir.exists():
            print("нет", zdir)
            continue
        x, y, split, M = T.load(ds, ev, 0, emb, False, 0)
        if (zdir / "sub.npy").exists():  # обучаемый DistilBERT: строки z -- подмножество текстов
            sub = np.load(zdir / "sub.npy")
            x, y, M = x[sub], y[sub], M[sub]
            n_fit, n_val = 12000, 2000
            split = {"fit": torch.arange(n_fit), "val": torch.arange(n_fit, n_fit + n_val),
                     "test": torch.arange(n_fit + n_val, len(sub))}
        Y = M.toarray().astype(np.float32)
        out = HERE / "results" / f"probe_{name}.jsonl"
        done = set()
        if out.exists():
            done = {(r["method"], r["lambda"], r["seed"], r["white"]) for r in map(json.loads, open(out))}
        te = split["test"]
        Yt = Y[te.numpy()]
        prev = {"ap": float(Yt.mean()), "macro_ap": float(Yt[:, Yt.sum(0) >= 5].mean(0).mean())}
        # вход тот же, что в базовом наборе (те же отложенные концепты) -- не пересчитывается
        same_input = any(t in name for t in ("_knob_", "_grid", "_dose_"))
        items = [] if same_input else [("input", 0.0, 0, F.normalize(x, dim=-1))]
        for f in sorted(zdir.glob("*_s*.npy")):
            method, lam, seed = f.stem.rsplit("_", 2)
            items.append((method, float(lam), int(seed[1:]), None, f))
        for it in items:
            method, lam, seed = it[:3]
            no_white = args.no_white or (args.white_input_only and method != "input")
            for white in ((False,) if no_white else (False, True)):
                if (method, lam, seed, white) in done:
                    continue
                try:
                    Z = it[3] if it[3] is not None else torch.tensor(np.load(it[4]).astype(np.float32))
                except (ValueError, OSError, EOFError):  # файл ещё пишется
                    continue
                X = whiten(Z.double(), split["fit"]) if white else rescale(Z.double(), split["fit"])
                m, S = probe(X.float().numpy(), Y, split, y, seed)
                if not white:  # оценки на тесте -- для парного бутстрапа
                    sdir = HERE / "results" / "scores" / name
                    sdir.mkdir(parents=True, exist_ok=True)
                    np.save(sdir / f"{method}_{lam:g}_s{seed}.npy", S.astype(np.float16))
                row = {"set": name, "method": method, "lambda": lam, "seed": seed, "white": white, **m,
                       "prevalence_ap": prev["ap"], "prevalence_macro_ap": prev["macro_ap"]}
                with open(out, "a") as fh:
                    fh.write(json.dumps(row) + "\n")
                print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)


if __name__ == "__main__":
    main()
