#!/usr/bin/env python
"""HiCEM, опыт Г3 (план experiments/PLAN_2026-11-10_checks.md): пересечение находок, бутстрэп с прогонами, P против Rtrue.
Только CPU.

Для каждого прогона R, Rtrue, P, C1 (по 3):
- сопоставленные понятия банка (их правило: лучший ROC-AUC в блоке > 0.7, как best_per_concept);
- значимые понятия: нуль «внутри родителя» (как hicem_checks2.py), K розыгрышей (по умолчанию 1000), полная семья,
  BH q = 0.1;
- по понятию «лучший ROC-AUC − среднее нуля».
Затем:
1. пересечение находок по всем парам прогонов двух вариантов (доля находок первого, которые есть и у второго; Жаккар) —
   для значимых и для сопоставленных; пары R–P, R–Rtrue, Rtrue–P, R–C1, P–C1;
2. двухуровневый бутстрэп (прогоны внутри варианта с возвращением, затем родители; 2000 выборок): разницы числа
   значимых, числа сопоставленных и средней «AUC − нуль» на общих понятиях — R − P, Rtrue − P, R − Rtrue, R − C1;
3. P против Rtrue по ROC-AUC табл. 1 (тест): средние по прогонам, разница, бутстрэп по прогонам.
Запуск из ~/hicem_repro/code/experiments: K=1000 NJ=4 python hicem_checks3.py ~/hicem_repro/configs/cub_R.yaml
Итог: ~/hicem_repro/results/hicem_checks3.json и сводка в stdout.
"""

import glob
import json
import os
import sys
from itertools import product
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

from experiment_utils import load_config, load_datasets
from hicem_checks import best_per_concept, load_results, matched_test_aucs

RES = Path(os.path.expanduser("~/hicem_repro/results"))
K = int(os.environ.get("K", "1000"))
NJ = int(os.environ.get("NJ", "4"))
VARIANTS = ("R", "Rtrue", "P", "C1")
N_BOOT = 2000
Q = 0.1


def bh_set(p, q=Q):
    """Индексы, прошедшие BH при уровне q."""
    o = np.argsort(p)
    m = len(p)
    ok = np.where(p[o] <= q * np.arange(1, m + 1) / m)[0]
    return set(o[: ok[-1] + 1].tolist()) if len(ok) else set()


def run_null(X, colsum, n_sub, sub_map, bank_cls, coarse_cls, parent, n_cls, ytr, seed):
    """Лучший ROC-AUC каждого понятия, сопоставленные, p по полной семье и «AUC − среднее нуля»."""
    rng = np.random.RandomState(seed)
    _, _, best_real, owned = best_per_concept(X, colsum, bank_cls[ytr], n_sub, sub_map)
    m = bank_cls.shape[1]
    ge, s_null, n_ok = np.zeros(m), np.zeros(m), np.zeros(m)
    for _ in range(K):
        bnn = bank_cls.copy()
        for j in range(m):
            pool = np.where(coarse_cls[:, parent[j]] == 1)[0] if j in parent else np.arange(n_cls)
            bnn[pool, j] = bank_cls[rng.permutation(pool), j]
        _, _, best_n, _ = best_per_concept(X, colsum, bnn[ytr], n_sub, sub_map)
        ok = ~np.isnan(best_n)
        ge[ok] += best_n[ok] >= np.nan_to_num(best_real[ok], nan=2)
        s_null[ok] += best_n[ok]
        n_ok[ok] += 1
    fam = np.where(~np.isnan(best_real))[0]
    p = (1 + ge[fam]) / (1 + K)
    sig = {int(fam[i]) for i in bh_set(p)}
    diff = best_real - np.where(n_ok > 0, s_null / np.maximum(n_ok, 1), np.nan)
    return {"sig": sorted(sig), "matched": sorted(int(j) for j in owned), "diff": diff.tolist(),
            "family": fam.tolist()}


def main():
    ds = load_datasets(load_config(sys.argv[1]))[0]
    ytr = ds.train_y.numpy().astype(int)
    ctr = ds.train_c.numpy()
    bank = ds.concept_bank.astype(np.float32)
    names = list(ds.concept_names)
    n_cls = ds.n_tasks
    first = np.array([np.where(ytr == k)[0][0] for k in range(n_cls)])
    bank_cls, coarse_cls = bank[first], ctr[first]
    parent = {}
    for top, members in enumerate(ds.sub_concept_map):
        for j in members:
            parent.setdefault(j, top)
    runs = {}
    for v in VARIANTS:
        for rd in sorted(glob.glob(str(RES / v / "*/"))):
            rd = Path(rd)
            if (rd / "results.yaml").exists():
                d = load_results(rd)
                if "enhanced_clip_vitl14_hicem_discovered_concept_auc" in d:
                    runs[f"{v}/{rd.name}"] = (v, rd, d)
    print("прогоны:", {v: sum(1 for r in runs.values() if r[0] == v) for v in VARIANTS}, flush=True)
    keys = list(runs)

    def job(i, r):
        v, rd, d = runs[r]
        X = np.load(rd / "discovered_concepts.npz")["discovered_concept_labels"].astype(np.float32)
        return r, run_null(X, X.sum(0), d["n_discovered_sub_concepts"], ds.sub_concept_map, bank_cls, coarse_cls,
                           parent, n_cls, ytr, 7000 + i)
    res = dict(Parallel(n_jobs=NJ)(delayed(job)(i, r) for i, r in enumerate(keys)))
    out = {"K": K, "runs": {r: {"n_sig": len(res[r]["sig"]), "n_matched": len(res[r]["matched"]),
                                "test_auc": float(runs[r][2]["enhanced_clip_vitl14_hicem_discovered_concept_auc"])}
                            for r in keys}}
    for r in keys:
        print(f"  {r:30s} значимых {len(res[r]['sig']):4d}, сопоставлено {len(res[r]['matched']):4d}, "
              f"ROC-AUC табл. 1 {out['runs'][r]['test_auc']:.4f}", flush=True)
    by_v = {v: [r for r in keys if runs[r][0] == v] for v in VARIANTS}

    # 1. пересечение находок
    out["overlap"] = {}
    print("\n1. пересечение находок (среднее по парам прогонов [мин; макс])", flush=True)
    for a, b in (("R", "P"), ("R", "Rtrue"), ("Rtrue", "P"), ("R", "C1"), ("P", "C1")):
        for kind in ("sig", "matched"):
            share, jac = [], []
            for ra, rb in product(by_v[a], by_v[b]):
                A, B = set(res[ra][kind]), set(res[rb][kind])
                if A:
                    share.append(len(A & B) / len(A))
                if A | B:
                    jac.append(len(A & B) / len(A | B))
            mmm = lambda x: [float(np.mean(x)), float(min(x)), float(max(x))] if x else [float("nan")] * 3  # noqa: E731
            out["overlap"][f"{a}-{b}|{kind}"] = {"share_of_first_in_second": mmm(share), "jaccard": mmm(jac)}
            sh, jc = mmm(share), mmm(jac)
            print(f"  {a:5s} → {b:5s} {kind:7s}: доля находок {a}, которые есть и у {b}, {sh[0]:.3f} "
                  f"[{sh[1]:.3f}; {sh[2]:.3f}]; Жаккар {jc[0]:.3f}", flush=True)

    # 2. двухуровневый бутстрэп
    aucs = {r: matched_test_aucs(runs[r][2]) for r in keys}
    common = sorted(set.intersection(*[set(a) for a in aucs.values() if a]))
    common_idx = np.array([names.index(s) for s in common if s in names])
    out["n_common"] = int(len(common_idx))
    m = bank.shape[1]
    ind = {r: {"sig": np.isin(np.arange(m), res[r]["sig"]).astype(float),
               "matched": np.isin(np.arange(m), res[r]["matched"]).astype(float),
               "diff": np.array(res[r]["diff"], float)} for r in keys}
    par = np.array([parent.get(j, -1) for j in range(m)])
    tops = np.unique(par)
    idx_by = {t: np.where(par == t)[0] for t in tops}
    cpar = par[common_idx]
    ctops = np.unique(cpar)
    cidx_by = {t: common_idx[cpar == t] for t in ctops}
    rng = np.random.RandomState(11)

    def stat(v, run_sel, cidx, allidx):
        rr = [by_v[v][i] for i in run_sel]
        return (np.mean([ind[r]["sig"][allidx].sum() for r in rr]),
                np.mean([ind[r]["matched"][allidx].sum() for r in rr]),
                np.mean([np.nanmean(ind[r]["diff"][cidx]) for r in rr]))
    real = {v: stat(v, range(len(by_v[v])), common_idx, np.arange(m)) for v in VARIANTS}
    boots = {v: [] for v in VARIANTS}
    for _ in range(N_BOOT):
        allidx = np.concatenate([idx_by[t] for t in rng.choice(tops, len(tops))])
        cidx = np.concatenate([cidx_by[t] for t in rng.choice(ctops, len(ctops))])
        for v in VARIANTS:
            boots[v].append(stat(v, rng.choice(len(by_v[v]), len(by_v[v])), cidx, allidx))
    out["boot"] = {}
    print(f"\n2. двухуровневый бутстрэп (прогоны, затем родители; {N_BOOT}); общих понятий {len(common_idx)}", flush=True)
    for a, b in (("R", "P"), ("Rtrue", "P"), ("R", "Rtrue"), ("R", "C1"), ("P", "C1")):
        for k, name in enumerate(("значимых", "сопоставлено", "AUC − нуль на общих")):
            d = np.array([x[k] for x in boots[a]]) - np.array([x[k] for x in boots[b]])
            est = real[a][k] - real[b][k]
            lo, hi = np.percentile(d, [2.5, 97.5])
            out["boot"][f"{a}-{b}|{name}"] = [float(est), float(lo), float(hi)]
            print(f"  {a:5s} − {b:5s} {name:20s}: {est:+.3f} [{lo:+.3f}; {hi:+.3f}]", flush=True)

    # 3. P против Rtrue по ROC-AUC табл. 1
    ta = {v: np.array([out["runs"][r]["test_auc"] for r in by_v[v]]) for v in VARIANTS}
    bs = [rng.choice(ta["P"], len(ta["P"])).mean() - rng.choice(ta["Rtrue"], len(ta["Rtrue"])).mean() for _ in range(N_BOOT)]
    out["P-Rtrue|test_auc"] = [float(ta["P"].mean() - ta["Rtrue"].mean()), float(np.percentile(bs, 2.5)),
                               float(np.percentile(bs, 97.5))]
    print(f"\n3. ROC-AUC табл. 1: P {ta['P'].round(4).tolist()}, Rtrue {ta['Rtrue'].round(4).tolist()}; P − Rtrue "
          f"{out['P-Rtrue|test_auc'][0]:+.4f} [{out['P-Rtrue|test_auc'][1]:+.4f}; {out['P-Rtrue|test_auc'][2]:+.4f}] "
          f"(бутстрэп по прогонам)", flush=True)
    out["per_run"] = res
    json.dump(out, open(RES / "hicem_checks3.json", "w"), ensure_ascii=False, indent=1)
    print("HICEM3_DONE", flush=True)


if __name__ == "__main__":
    main()
