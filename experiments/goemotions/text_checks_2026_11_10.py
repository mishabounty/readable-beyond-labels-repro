#!/usr/bin/env python
"""Досчёты на текстах по рецензиям на v9 (план experiments/PLAN_2026-11-10_texts.md, записан до счёта).

Т1 — парные интервалы смысла против годных пустышек (bce − hyp, bce − km, bce − rowperm) по готовым результатам
     probe_ridge (ext) и cond_probe_scaled (ext); границы d̄ ± 2.776·s/√5 и бутстрэп по сидам.
Т2 — таблица свойств меток: настоящие A, rand, rowperm, hyp, km (пустышки — по 5 сидам): доли, число понятий на текст,
     средний |φ|, согласованность с входом (ridge: исходный эмбеддинг → метки), утечка в B (ridge: метки → B),
     насколько выучены (ridge: z своей модели → свои метки), P(есть B | есть A) и P(есть B | нет A).
Т3 — «доза — ответ»: Δ_j = AP_bce(j) − AP_пустышки(j) по 14 эмоциям B против max|φ(B_j, A)| и числа эмоций той же
     группы Экмана в A (google-research/goemotions/data/ekman_mapping.json); Спирмен и бутстрэп по эмоциям.
Пустышки пересоздаются так же, как в text_finetune.py (функции перенесены сюда без изменений, чтобы не тянуть
зависимости обучения); rand — train.load(..., random_concepts=True, seed).
Запуск: RIDGE_GRID=ext python text_checks_2026_11_10.py; итог: results/text_checks_2026_11_10.json и сводка в stdout.
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from scipy import stats

import train as T
from cond_probe_scaled import load_set
from probe_eval import _platt, rescale
from probe_ridge import ALPHAS, ap_columns, ridge_path

HERE = Path(__file__).resolve().parent
NAME = "finetune_goemotions_length_v2"
SEEDS = [0, 1, 2, 3, 4]
PLACEBOS = ("rand", "rowperm", "hyp", "km")
EKMAN = {"anger": ["anger", "annoyance", "disapproval"], "disgust": ["disgust"], "fear": ["fear", "nervousness"],
         "joy": ["joy", "amusement", "approval", "excitement", "gratitude", "love", "optimism", "relief", "pride",
                 "admiration", "desire", "caring"],
         "sadness": ["sadness", "disappointment", "embarrassment", "grief", "remorse"],
         "surprise": ["surprise", "realization", "confusion", "curiosity"]}


def hyperplane_concepts(x, M_A, fit, seed):
    """Как в text_finetune.py: случайные гиперплоскости входа с теми же долями положительных, что у концептов A."""
    rng = np.random.default_rng(20_000 + seed)
    s = F.normalize(x, dim=-1).numpy() @ rng.standard_normal((x.shape[1], M_A.shape[1]))
    p = np.asarray(M_A[fit.numpy()].mean(0)).ravel()
    thr = np.array([np.quantile(s[fit.numpy(), j], 1 - p[j]) for j in range(len(p))])
    return sp.csr_matrix((s > thr).astype(np.int8))


def kmeans_concepts(x, M_A, fit, seed):
    """Как в text_finetune.py: k-means на исходном эмбеддинге, «концепт» j — косинус к центру j выше квантиля."""
    from sklearn.cluster import KMeans
    xn, f = F.normalize(x, dim=-1).numpy(), fit.numpy()
    km = KMeans(M_A.shape[1], n_init=3, random_state=seed).fit(xn[f])
    c = km.cluster_centers_ / np.linalg.norm(km.cluster_centers_, axis=1, keepdims=True)
    s = xn @ c.T
    p = np.asarray(M_A[f].mean(0)).ravel()
    thr = np.array([np.quantile(s[f, j], 1 - p[j]) for j in range(len(p))])
    return sp.csr_matrix((s > thr).astype(np.int8))


def rowperm_concepts(M_A, split, seed):
    """Как в text_finetune.py: строки меток A перемешаны между текстами внутри каждой части."""
    perm = np.arange(M_A.shape[0])
    rngp = np.random.RandomState(30_000 + seed)
    for part in ("fit", "val", "test"):
        idx = split[part].numpy()
        perm[idx] = idx[rngp.permutation(len(idx))]
    return M_A[perm]


def probe_cols(X, Y, split):
    """Как probe_ridge.probe (alpha по val для каждого столбца, Платт по val), но возвращает AP каждого столбца на
    test (столбцы с ≥ 5 положительными, как в train.concept_metrics; прочие — nan) и macro-AP."""
    dev = T.DEV if X.shape[1] <= 1024 else "cpu"
    fit, val, te = (split[k].numpy() for k in ("fit", "val", "test"))
    Xd = torch.tensor(X, device=dev, dtype=torch.float64)
    Yd = torch.tensor(Y, device=dev, dtype=torch.float64)
    m = Y.shape[1]
    best = torch.full((m,), -1.0, device=dev, dtype=torch.float64)
    S = torch.zeros(len(te), m, device=dev, dtype=torch.float64)
    Sv = torch.zeros(len(val), m, device=dev, dtype=torch.float64)
    for sv, st in ridge_path(Xd[fit], Yd[fit], [Xd[val], Xd[te]]):
        apv = ap_columns(sv, Yd[val]).nan_to_num(0.0)
        better = apv > best
        best[better] = apv[better]
        S[:, better], Sv[:, better] = st[:, better], sv[:, better]
    S, Sv = S.cpu().numpy(), Sv.cpu().numpy()
    for j in range(m):
        S[:, j] = _platt(Sv[:, j], Y[val, j], S[:, j])
    ap = ap_columns(torch.tensor(S), torch.tensor(Y[te], dtype=torch.float64)).numpy()
    ok = Y[te].sum(0) >= 5
    ap = np.where(ok, ap, np.nan)
    return ap, float(np.nanmean(ap))


def phi_matrix(P, Q):
    """|φ| между столбцами бинарных матриц P (n × p) и Q (n × q)."""
    P, Q = P.astype(np.float64), Q.astype(np.float64)
    Pc, Qc = P - P.mean(0), Q - Q.mean(0)
    den = np.sqrt((Pc ** 2).sum(0))[:, None] * np.sqrt((Qc ** 2).sum(0))[None, :]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.abs((Pc.T @ Qc) / den)


def interval(d):
    d = np.asarray(d, float)
    h = 2.776 * d.std(ddof=1) / np.sqrt(len(d))
    rng = np.random.RandomState(0)
    bs = [rng.choice(d, len(d)).mean() for _ in range(5000)]
    return {"mean": float(d.mean()), "lo": float(d.mean() - h), "hi": float(d.mean() + h),
            "boot_lo": float(np.percentile(bs, 2.5)), "boot_hi": float(np.percentile(bs, 97.5)),
            "per_seed": [round(float(v), 4) for v in d]}


def main():
    out = {"T1": {}, "T2": {}, "T3": {}, "check": {}}
    # ---------------- Т1: готовые результаты
    for tag, fn in (("ridge", "probe_ridge_ext"), ("known_A", "cond_probe_scaled_ext")):
        g = defaultdict(dict)
        for r in map(json.loads, open(HERE / "results" / f"{fn}_{NAME}.jsonl")):
            g[r["method"]][r["seed"]] = r["macro_ap"]
        for b in ("hyp", "km", "rowperm", "rand", "task"):
            ks = sorted(set(g["bce"]) & set(g[b]))
            out["T1"][f"{tag}|bce-{b}"] = interval([g["bce"][k] - g[b][k] for k in ks])
        for b in ("hyp", "km", "rowperm", "rand"):
            ks = sorted(set(g[b]) & set(g["task"]))
            out["T1"][f"{tag}|{b}-task"] = interval([g[b][k] - g["task"][k] for k in ks])
        out["T1"][f"{tag}|means"] = {m: float(np.mean(list(v.values()))) for m, v in g.items()}
    print("Т1: парные разницы по 5 сидам (d̄ [d̄ ± 2.776·s/√5]; бутстрэп по сидам)")
    for k, v in out["T1"].items():
        if "|means" in k:
            continue
        print(f"  {k:22s} {v['mean']:+.4f} [{v['lo']:+.4f}; {v['hi']:+.4f}]  бутстрэп [{v['boot_lo']:+.4f}; "
              f"{v['boot_hi']:+.4f}]  по сидам {v['per_seed']}", flush=True)

    # ---------------- данные
    zdir, x, y, split, Y, MA = load_set(NAME)
    fit = split["fit"]
    nf = fit.numpy()
    A_true = MA.toarray().astype(np.float32)
    namesA = json.loads((HERE / "data" / "goemotions" / "concepts_emotionA.json").read_text())
    namesB = json.loads((HERE / "data" / "goemotions" / "concepts_emotionB.json").read_text())
    X_in = rescale(F.normalize(x, dim=-1).double(), fit).float().numpy()
    labels = {"A": {0: A_true}}
    for seed in SEEDS:
        labels.setdefault("rand", {})[seed] = T.load("goemotions", "emotionA", 0, "mqa", True, seed)[3][
            np.load(zdir / "sub.npy")].toarray().astype(np.float32)
        labels.setdefault("rowperm", {})[seed] = rowperm_concepts(MA, split, seed).toarray().astype(np.float32)
        labels.setdefault("hyp", {})[seed] = hyperplane_concepts(x, MA, fit, seed).toarray().astype(np.float32)
        labels.setdefault("km", {})[seed] = kmeans_concepts(x, MA, fit, seed).toarray().astype(np.float32)
    chance_B = float(Y[split["test"].numpy()].mean())
    hasB = Y.sum(1) > 0

    # ---------------- Т2: свойства меток
    print(f"\nТ2: свойства меток (доля «да» у B в тесте, то есть AP наугад ≈ {chance_B:.4f})", flush=True)
    for kind, per_seed in labels.items():
        rows = []
        for seed, L in per_seed.items():
            rate = L[nf].mean(0)
            ph = phi_matrix(L[nf], L[nf])
            iu = np.triu_indices(L.shape[1], 1)
            hasA = L.sum(1) > 0
            _, pred_in = probe_cols(X_in, L, split)
            _, leak = probe_cols(rescale(torch.tensor(L, dtype=torch.float64), fit).float().numpy(), Y, split)
            zmeth = "bce" if kind == "A" else kind
            zfile = zdir / f"{zmeth}_1_s{seed if kind != 'A' else 0}.npy"
            learn = None
            if kind == "A":
                learn = [probe_cols(rescale(torch.tensor(np.load(zdir / f"bce_1_s{s}.npy").astype(np.float64)),
                                            fit).float().numpy(), L, split)[1] for s in SEEDS]
                learn = float(np.mean(learn))
            elif zfile.exists():
                Z = rescale(torch.tensor(np.load(zfile).astype(np.float64)), fit).float().numpy()
                learn = probe_cols(Z, L, split)[1]
            rows.append({"seed": seed, "rate_mean": float(rate.mean()), "rate_min": float(rate.min()),
                         "rate_max": float(rate.max()), "per_text": float(L[nf].sum(1).mean()),
                         "phi_mean": float(np.nanmean(ph[iu])), "pred_from_input_ap": pred_in,
                         "chance_own": float(L[split["test"].numpy()].mean()),
                         "leak_B_ap": leak, "learned_ap": learn,
                         "P_B_given_A": float(hasB[hasA].mean()) if hasA.any() else None,
                         "P_B_given_noA": float(hasB[~hasA].mean())})
        out["T2"][kind] = rows
        agg = {k: (np.mean([r[k] for r in rows if r[k] is not None]) if any(r[k] is not None for r in rows) else None,
                   min((r[k] for r in rows if r[k] is not None), default=None),
                   max((r[k] for r in rows if r[k] is not None), default=None))
               for k in rows[0] if k != "seed"}
        print(f"  {kind:8s} " + "; ".join(f"{k} {v[0]:.4f}" + (f" ({v[1]:.4f}–{v[2]:.4f})" if v[1] != v[2] else "")
                                        for k, v in agg.items() if v[0] is not None), flush=True)

    # ---------------- Т3: доза — ответ
    phiBA = phi_matrix(Y[nf], A_true[nf])  # 14 × 13
    maxphi = np.nanmax(phiBA, 1)
    group = {e: g for g, es in EKMAN.items() for e in es}
    same_group_A = np.array([sum(group[a] == group[b] for a in namesA) for b in namesB])
    ap_by = {}
    stored = {r["seed"]: r["macro_ap"] for r in map(json.loads, open(HERE / "results" /
                                                                       f"probe_ridge_ext_{NAME}.jsonl"))
              if r["method"] == "bce"}
    for meth in ("bce", "hyp", "km", "rowperm", "task"):
        per = []
        for seed in SEEDS:
            Z = rescale(torch.tensor(np.load(zdir / f"{meth}_{'0' if meth == 'task' else '1'}_s{seed}.npy")
                                     .astype(np.float64)), fit).float().numpy()
            ap, mac = probe_cols(Z, Y, split)
            per.append(ap)
            if meth == "bce":
                out["check"][f"bce_s{seed}"] = {"recomputed": mac, "stored": stored.get(seed)}
        ap_by[meth] = np.nanmean(np.array(per), 0)
    print("\nсверка macro-AP bce с сохранёнными: " + ", ".join(
        f"s{k[-1]} {v['recomputed']:.4f}/{v['stored']:.4f}" for k, v in out["check"].items()), flush=True)
    rng = np.random.RandomState(0)
    print("\nТ3: доза — ответ по 14 эмоциям B (Спирмен; бутстрэп по эмоциям, 2000)")
    for b in ("hyp", "km", "rowperm"):
        d = ap_by["bce"] - ap_by[b]
        ok = ~np.isnan(d)
        res = {"delta_per_emotion": dict(zip(namesB, [None if np.isnan(v) else round(float(v), 4) for v in d]))}
        for mname, mval in (("maxphi", maxphi), ("same_group_A", same_group_A)):
            rho = stats.spearmanr(mval[ok], d[ok])[0]
            idx = np.where(ok)[0]
            bs = []
            for _ in range(2000):
                s = rng.choice(idx, len(idx))
                if len(set(mval[s])) > 1:
                    bs.append(stats.spearmanr(mval[s], d[s])[0])
            res[mname] = {"rho": float(rho), "lo": float(np.nanpercentile(bs, 2.5)), "hi": float(np.nanpercentile(bs, 97.5))}
            print(f"  bce − {b:7s} против {mname:12s}: ρ = {rho:+.3f} [{res[mname]['lo']:+.3f}; {res[mname]['hi']:+.3f}]",
                  flush=True)
        out["T3"][b] = res
    out["T3"]["maxphi"] = dict(zip(namesB, [round(float(v), 4) for v in maxphi]))
    out["T3"]["same_group_A"] = dict(zip(namesB, same_group_A.tolist()))
    print("  max|φ| и число эмоций той же группы в A: " + ", ".join(
        f"{b} {maxphi[i]:.3f}/{same_group_A[i]}" for i, b in enumerate(namesB)), flush=True)
    print("  Δ_j (bce − hyp): " + ", ".join(f"{b} {v:+.3f}" for b, v in zip(namesB, ap_by["bce"] - ap_by["hyp"])),
          flush=True)
    json.dump(out, open(HERE / "results" / "text_checks_2026_11_10.json", "w"), ensure_ascii=False, indent=1)
    print("TEXT_CHECKS_DONE", flush=True)


if __name__ == "__main__":
    main()
