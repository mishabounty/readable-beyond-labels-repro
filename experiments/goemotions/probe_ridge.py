#!/usr/bin/env python
"""Линейная читаемость в замкнутом виде: ridge-проба (МНК с L2) на сохранённых z -- для свипов, где логистическая
регрессия до сходимости (probe_eval.py) слишком дорога; на основных постановках обе пробы считаются и сверяются.

Признаки центрируются и делятся на общий скаляр (как в probe_eval.py) или отбеливаются; по fit одно спектральное
разложение X^T X = V diag(l) V^T, затем для всех концептов и всех alpha сразу w = V diag(1 / (l + alpha)) V^T X^T y,
свободный член -- среднее y. alpha = mean(l) * {1e-4 .. 1e2}, выбирается для каждого концепта по AP на валидации;
оценки калибруются по Платту на валидации. 200 меток: единое alpha по 4-кратной кросс-валидации (macro-AP),
калибровка по оценкам вне фолда. Метрики -- как в probe_eval.py.
Результат: results/probe_ridge_<набор>.jsonl.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import train as T
from collapse_check import within_class_auc
from probe_eval import _platt, rescale, spec, whiten

HERE = Path(__file__).resolve().parent
# RIDGE_GRID=ext (26.10): сетка штрафа расширена вниз до 1e-8, т.к. у сжатых z выбор упирался в 1e-4;
# результаты пишутся в отдельные файлы с суффиксом _ext, исходные не трогаются
GRID = os.environ.get("RIDGE_GRID", "")
ALPHAS = ([1e-8, 1e-7, 1e-6, 1e-5] if GRID == "ext" else []) + [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
SUFFIX = "_ext" if GRID == "ext" else ""


def ap_columns(S, Y):
    o = S.argsort(0, descending=True)
    Ys = Y.gather(0, o)
    prec = Ys.cumsum(0) / torch.arange(1, len(S) + 1, device=S.device)[:, None]
    pos = Ys.sum(0)
    ap = (prec * Ys).sum(0) / pos.clamp_min(1)
    ap[pos == 0] = float("nan")
    return ap


def ridge_path(Xtr, Ytr, Xev_list):
    """Для каждого alpha -- оценки на каждой матрице из Xev_list (все концепты сразу)."""
    mu = Xtr.mean(0)
    Xc = Xtr - mu
    lam, V = torch.linalg.eigh(Xc.t() @ Xc)
    lam = lam.clamp_min(0)
    P = V.t() @ (Xc.t() @ (Ytr - Ytr.mean(0)))  # d x m
    proj = [(Xe - mu) @ V for Xe in Xev_list]
    scale = lam.mean()
    out = []
    for a in ALPHAS:
        coef = P / (lam + a * scale)[:, None]
        out.append([Pe @ coef + Ytr.mean(0) for Pe in proj])
    return out


def macro_ap(S, Y):
    return T.concept_metrics(torch.as_tensor(S, dtype=torch.float32), torch.as_tensor(Y, dtype=torch.float32))["macro_ap"]


def probe(X, Y, split, y_task, seed):
    dev = T.DEV if X.shape[1] <= 1024 else "cpu"  # вход TF-IDF (4096 признаков) в float64 не помещается в GPU
    fit, val, te = (split[k].numpy() for k in ("fit", "val", "test"))
    Xd = torch.tensor(X, device=dev, dtype=torch.float64)
    Yd = torch.tensor(Y, device=dev, dtype=torch.float64)
    m = Y.shape[1]
    paths = ridge_path(Xd[fit], Yd[fit], [Xd[val], Xd[te]])
    best = torch.full((m,), -1.0, device=dev, dtype=torch.float64)
    bestA = torch.zeros(m, device=dev, dtype=torch.float64)
    S = torch.zeros(len(te), m, device=dev, dtype=torch.float64)
    Sv = torch.zeros(len(val), m, device=dev, dtype=torch.float64)
    for a, (sv, st) in zip(ALPHAS, paths):
        apv = ap_columns(sv, Yd[val]).nan_to_num(0.0)
        better = apv > best
        best[better], bestA[better] = apv[better], a
        S[:, better], Sv[:, better] = st[:, better], sv[:, better]
    S, Sv = S.cpu().numpy(), Sv.cpu().numpy()
    for j in range(m):
        S[:, j] = _platt(Sv[:, j], Y[val, j], S[:, j])
    met = T.concept_metrics(torch.tensor(S, dtype=torch.float32), torch.tensor(Y[te], dtype=torch.float32))
    out = {"alpha_median": float(bestA.median()), "ap": met["ap"], "macro_ap": met["macro_ap"],
           "auc_within": within_class_auc(torch.tensor(S), torch.tensor(Y[te]), y_task[torch.as_tensor(te)])}
    g = torch.Generator().manual_seed(1000 + seed)
    few = fit[torch.randperm(len(fit), generator=g)[:200].numpy()]
    folds = np.arange(len(few)) % 4
    oof = np.zeros((len(ALPHAS), len(few), m))
    for f in range(4):
        tr, va = few[folds != f], few[folds == f]
        for k, (s,) in enumerate(ridge_path(Xd[tr], Yd[tr], [Xd[va]])):
            oof[k][folds == f] = s.cpu().numpy()
    cv = [np.mean([macro_ap(oof[k][folds == f], Y[few][folds == f]) for f in range(4)]) for k in range(len(ALPHAS))]
    k = int(np.argmax(cv))
    S2 = ridge_path(Xd[few], Yd[few], [Xd[te]])[k][0].cpu().numpy()
    for j in range(m):
        S2[:, j] = _platt(oof[k][:, j], Y[few, j], S2[:, j])
    m2 = T.concept_metrics(torch.tensor(S2, dtype=torch.float32), torch.tensor(Y[te], dtype=torch.float32))
    out.update(ap_200=m2["ap"], macro_ap_200=m2["macro_ap"], alpha_200=ALPHAS[k])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sets", nargs="+")
    ap.add_argument("--no-white", action="store_true")
    args = ap.parse_args()
    for name in args.sets:
        ds, ev, emb = spec(name)
        zdir = HERE / "results" / "z" / name
        if not zdir.exists():
            continue
        x, y, split, M = T.load(ds, ev, 0, emb, False, 0)
        if (zdir / "sub.npy").exists():
            sub = np.load(zdir / "sub.npy")
            x, y, M = x[sub], y[sub], M[sub]
            nf, nv = (int(v) for v in np.load(zdir / "sub_split.npz").values()) if (zdir / "sub_split.npz").exists() \
                else (12000, 2000)
            split = {"fit": torch.arange(nf), "val": torch.arange(nf, nf + nv), "test": torch.arange(nf + nv, len(sub))}
        Y = M.toarray().astype(np.float32)
        out = HERE / "results" / f"probe_ridge{SUFFIX}_{name}.jsonl"
        done = {(r["method"], r["lambda"], r["seed"], r["white"]) for r in map(json.loads, open(out))} \
            if out.exists() else set()
        same_input = any(t in name for t in ("_knob_", "_grid", "_dose_"))
        items = [] if same_input else [("input", 0.0, 0, F.normalize(x, dim=-1))]
        for f in sorted(zdir.glob("*_s*.npy")):
            method, lam, seed = f.stem.rsplit("_", 2)
            items.append((method, float(lam), int(seed[1:]), None, f))
        for it in items:
            method, lam, seed = it[:3]
            for white in ((False,) if args.no_white else (False, True)):
                if (method, lam, seed, white) in done:
                    continue
                try:
                    Z = it[3] if it[3] is not None else torch.tensor(np.load(it[4]).astype(np.float32))
                except (ValueError, OSError, EOFError):  # файл ещё пишется
                    continue
                X = whiten(Z.double(), split["fit"]) if white else rescale(Z.double(), split["fit"])
                m = probe(X.float().numpy(), Y, split, y, seed)
                row = {"set": name, "method": method, "lambda": lam, "seed": seed, "white": white, **m}
                with open(out, "a") as fh:
                    fh.write(json.dumps(row) + "\n")
                print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
