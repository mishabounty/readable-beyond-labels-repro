#!/usr/bin/env python
"""Проверки HiCEM по рецензиям на v8 (план experiments/PLAN_2026-11-02_checks.md, N7). Только CPU.

1. p по ПОЛНОЙ семье: для каждого понятия банка (у которого есть положительные и отрицательные обучающие фото и блок
   кандидатов) лучший ROC-AUC в его блоке сравнивается с нулём «внутри родителя» (метки понятия перемешаны между
   видами внутри видов родителя, как в null_match.py), K розыгрышей (по умолчанию 1000). BH q = 0.1 по одной и той же
   полной семье у всех вариантов; «значимых» — число понятий семьи, прошедших BH (сопоставлено или нет).
2. B2 для Rnull: пол «родитель как оценка» и запас HiCEM над ним по собственному случайному банку прогона
   (банк восстанавливается по зерну, как в run_ctrl.py: ALPHA — зерно 0, BRAVO — 2, CHARLIE — 1).
3. На понятиях, сопоставленных во всех прогонах R, Rtrue, P, C1 (как B1): разница «лучший ROC-AUC − среднее нуля» по
   понятию (обучающие фото), среднее по прогонам варианта; интервалы бутстрэпом по родителям (2000 выборок);
   парные разницы R − P, R − C1, Rtrue − C1.
Запуск из ~/hicem_repro/code/experiments: K=1000 python hicem_checks2.py ~/hicem_repro/configs/cub_R.yaml
Итог: ~/hicem_repro/results/hicem_checks2.json и сводка в stdout.
"""

import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

from experiment_utils import load_config, load_datasets
from hicem_checks import bh, best_per_concept, load_results, matched_test_aucs

RES = Path(os.path.expanduser("~/hicem_repro/results"))
K = int(os.environ.get("K", "1000"))
VARIANTS = ("R", "Rtrue", "P", "C1", "Rnull")
RNULL_SEED = {"cub-ALPHA-ALPHA-ALPHA": 0, "cub-ALPHA-ALPHA-BRAVO": 2, "cub-ALPHA-ALPHA-CHARLIE": 1}
N_BOOT = 2000


def null_bank(bank_cls, coarse_cls, parent, n_cls, seed):
    """Случайный банк Rnull ровно как в run_ctrl.py (режим null_bank)."""
    rng = np.random.RandomState(seed)
    bnull = bank_cls.copy()
    for j in range(bank_cls.shape[1]):
        pool = np.where(coarse_cls[:, parent[j]] == 1)[0] if j in parent else np.arange(n_cls)
        bnull[pool, j] = bank_cls[rng.permutation(pool), j]
    return bnull


def main():
    ds = load_datasets(load_config(sys.argv[1]))[0]
    ytr, yte = ds.train_y.numpy().astype(int), ds.test_y.numpy().astype(int)
    ctr, cte = ds.train_c.numpy(), ds.test_c.numpy()
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
    out = {"K": K, "family": {}, "B2_Rnull": {}, "common": {}}
    # общие понятия (как B1, без Rnull: у него другой банк)
    aucs = {r: matched_test_aucs(d) for r, (v, rd, d) in runs.items() if v != "Rnull"}
    common = sorted(set.intersection(*[set(a) for a in aucs.values() if a]))
    common_idx = [names.index(s) for s in common if s in names]
    out["n_common"] = len(common_idx)

    # 2. B2 для Rnull
    for r, (v, rd, d) in runs.items():
        if v != "Rnull":
            continue
        bn = null_bank(bank_cls, coarse_cls, parent, n_cls, RNULL_SEED[rd.name])
        gt = bn[yte]
        a = matched_test_aucs(d)
        floors, margins = [], []
        for s, val in a.items():
            j = names.index(s)
            y = gt[:, j]
            P, N = y.sum(), len(y) - y.sum()
            if j not in parent or P == 0 or N == 0:
                continue
            sc = cte[:, parent[j]].astype(float)
            fl = 0.5 + 0.5 * ((sc * y).sum() / P - (sc * (1 - y)).sum() / N)
            floors.append(fl), margins.append(val - fl)
        out["B2_Rnull"][r] = {"n": len(floors), "floor_mean": float(np.mean(floors)), "margin_mean": float(np.mean(margins)),
                              "test_auc_mean": float(np.mean(list(a.values())))}
        print(f"B2 {r:30s} пол родителя {np.mean(floors):.4f}; ROC-AUC {np.mean(list(a.values())):.4f}; "
              f"запас над полом {np.mean(margins):+.4f} ({len(floors)} понятий)", flush=True)

    # 1 и 3. нуль «внутри родителя», полная семья
    rng = np.random.RandomState(0)
    per_concept = {}
    for r, (v, rd, d) in runs.items():
        X = np.load(rd / "discovered_concepts.npz")["discovered_concept_labels"].astype(np.float32)
        colsum = X.sum(0)
        n_sub = d["n_discovered_sub_concepts"]
        bcls = null_bank(bank_cls, coarse_cls, parent, n_cls, RNULL_SEED[rd.name]) if v == "Rnull" else bank_cls
        cnt, _, best_real, _ = best_per_concept(X, colsum, bcls[ytr], n_sub, ds.sub_concept_map)
        ge = np.zeros(bank.shape[1])
        sum_null = np.zeros(bank.shape[1])
        n_ok = np.zeros(bank.shape[1])
        for _ in range(K):
            bnn = bcls.copy()
            for j in range(bank.shape[1]):
                pool = np.where(coarse_cls[:, parent[j]] == 1)[0] if j in parent else np.arange(n_cls)
                bnn[pool, j] = bcls[rng.permutation(pool), j]
            _, _, best_n, _ = best_per_concept(X, colsum, bnn[ytr], n_sub, ds.sub_concept_map)
            ok = ~np.isnan(best_n)
            ge[ok] += best_n[ok] >= np.nan_to_num(best_real[ok], nan=2)
            sum_null[ok] += best_n[ok]
            n_ok[ok] += 1
        fam = ~np.isnan(best_real)
        p = (1 + ge[fam]) / (1 + K)
        n_sig = bh(p)
        out["family"][r] = {"family_size": int(fam.sum()), "matched": cnt, "n_sig_bh_q0.1": n_sig,
                            "min_p": float(p.min())}
        per_concept[r] = best_real - np.where(n_ok > 0, sum_null / np.maximum(n_ok, 1), np.nan)
        print(f"семья {r:30s}: понятий в семье {int(fam.sum())}, сопоставлено {cnt}, значимых при BH q = 0.1 "
              f"(полная семья, K = {K}): {n_sig}", flush=True)
        json.dump(out, open(RES / "hicem_checks2.json", "w"), ensure_ascii=False, indent=1)

    # 3. разница с нулём на общих понятиях, бутстрэп по родителям
    var_mean = {}
    for v in ("R", "Rtrue", "P", "C1"):
        rr = [per_concept[r][common_idx] for r in per_concept if r.startswith(v + "/")]
        if rr:
            var_mean[v] = np.nanmean(np.stack(rr), 0)
    par = np.array([parent.get(j, -1) for j in common_idx])
    tops = np.unique(par)
    rngb = np.random.RandomState(1)
    idx_by = {t: np.where(par == t)[0] for t in tops}
    draws = [np.concatenate([idx_by[t] for t in rngb.choice(tops, len(tops))]) for _ in range(N_BOOT)]

    def ci(vec):
        bs = [np.nanmean(vec[ix]) for ix in draws]
        return [float(np.nanmean(vec)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    print(f"\nна {len(common_idx)} общих понятиях: лучший ROC-AUC (обучающие фото) − среднее нуля «внутри родителя», "
          f"бутстрэп по {len(tops)} родителям:", flush=True)
    for v, vec in var_mean.items():
        out["common"][v] = ci(vec)
        print(f"  {v:6s}: {out['common'][v][0]:+.4f} [{out['common'][v][1]:+.4f}; {out['common'][v][2]:+.4f}]", flush=True)
    for a, b in (("R", "P"), ("R", "C1"), ("Rtrue", "C1"), ("R", "Rtrue")):
        if a in var_mean and b in var_mean:
            out["common"][f"{a}-{b}"] = ci(var_mean[a] - var_mean[b])
            c = out["common"][f"{a}-{b}"]
            print(f"  {a} − {b}: {c[0]:+.4f} [{c[1]:+.4f}; {c[2]:+.4f}]", flush=True)
    json.dump(out, open(RES / "hicem_checks2.json", "w"), ensure_ascii=False, indent=1)
    print("HICEM2_DONE", flush=True)


if __name__ == "__main__":
    main()
