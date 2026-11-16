#!/usr/bin/env python
"""Проверки HiCEM по рецензиям 30.10 (план experiments/PLAN_2026-10-30_checks.md, пункты B1–B3). Только CPU.

B1 — ROC-AUC итоговой HiCEM на тесте на ПЕРЕСЕЧЕНИИ понятий банка, сопоставленных во всех вариантах (R, Rtrue, P, C1;
     все прогоны) — без отбора выживших.
B2 — пол «родитель как оценка»: ROC-AUC предсказателя «1 на всех тестовых фото, где родительское грубое понятие = 1»
     для каждого сопоставленного понятия банка; запас HiCEM над этим полом.
B3 — нулевые модели при K розыгрышах (по умолчанию 200) для кандидатов каждого прогона:
     within  — метки каждого понятия банка перемешаны между видами внутри видов родителя (как в null_match.py);
     parent  — одна перестановка видов на родителя (общая для всех понятий родителя);
     image   — метки перемешаны между ФОТО внутри фото родителя (рвёт видовую структуру).
     Сообщается охват сверх нуля (настоящее число сопоставленных минус среднее при нуле), поконцептные
     перестановочные p (доля розыгрышей, где лучший ROC-AUC понятия в его блоке ≥ настоящего) и число значимых при
     BH-FDR q = 0.1.
Запуск из ~/hicem_repro/code/experiments: K=200 python hicem_checks.py ~/hicem_repro/configs/cub_R.yaml
Итог: ~/hicem_repro/results/$OUT_NAME (по умолчанию hicem_checks.json) и сводка в stdout.
"""

import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import yaml

from experiment_utils import load_config, load_datasets

RES = Path(os.path.expanduser("~/hicem_repro/results"))
K = int(os.environ.get("K", "200"))
OUT_NAME = os.environ.get("OUT_NAME", "hicem_checks.json")  # окончательный прогон после повторов: hicem_checks_final.json
VARIANTS = ("R", "Rtrue", "P", "C1")


def load_results(rd):
    d = {}
    for doc in yaml.safe_load_all(open(rd / "results.yaml")):
        if isinstance(doc, dict):
            d.update(doc)
    return d


def matched_test_aucs(d):
    roc = np.array(d["discovered_concept_roc_aucs"], float)
    m = roc > 0
    sem = [s for s, ok in zip(d["discovered_concept_semantics"], m) if ok]
    aucs = d.get("enhanced_clip_vitl14_hicem_discovered_concept_aucs", [])
    return dict(zip(sem, aucs)) if len(aucs) == len(sem) else {}


def best_per_concept(X, colsum, Y, n_sub, sub_map, thr=0.7):
    """Их правило сопоставления (как null_match.match) + лучший ROC-AUC каждого понятия банка в его блоке."""
    n, m = X.shape
    starts = np.concatenate([[0], np.cumsum(n_sub)]).astype(int)
    P = Y.sum(0)
    N = n - P
    TP = Y.T @ X
    FP = colsum[None, :] - TP
    with np.errstate(divide="ignore", invalid="ignore"):
        AUC = 0.5 + 0.5 * (TP / P[:, None] - FP / N[:, None])
    best_auc = np.zeros(m)
    owner = np.full(m, -1)
    best_j = np.full(Y.shape[1], np.nan)
    for j in range(Y.shape[1]):
        if P[j] == 0 or N[j] == 0:
            continue
        lo, hi = 0, m
        for top, ns in enumerate(n_sub):
            if j in sub_map[top]:
                lo, hi = starts[top], starts[top] + ns
                break
        if hi <= lo:
            continue
        a = AUC[j, lo:hi]
        k = int(np.argmax(a))
        best_j[j] = a[k]
        if a[k] > thr and a[k] > best_auc[lo + k]:
            best_auc[lo + k] = a[k]
            owner[lo + k] = j
    ok = owner >= 0
    return int(ok.sum()), (float(best_auc[ok].mean()) if ok.any() else float("nan")), best_j, set(owner[ok].tolist())


def bh(pvals, q=0.1):
    p = np.sort(np.asarray(pvals))
    m = len(p)
    ok = np.where(p <= q * np.arange(1, m + 1) / m)[0]
    return int(ok[-1] + 1) if len(ok) else 0


def main():
    ds = load_datasets(load_config(sys.argv[1]))[0]
    ytr, yte = ds.train_y.numpy().astype(int), ds.test_y.numpy().astype(int)
    ctr, cte = ds.train_c.numpy(), ds.test_c.numpy()
    bank = ds.concept_bank.astype(np.float32)
    test_gt = ds.concept_test_ground_truth.astype(np.float32)
    names = list(ds.concept_names)
    n_cls = ds.n_tasks
    first = np.array([np.where(ytr == k)[0][0] for k in range(n_cls)])
    bank_cls, coarse_cls = bank[first], ctr[first]
    parent = {}
    for top, members in enumerate(ds.sub_concept_map):
        for j in members:
            parent.setdefault(j, top)
    out = {"K": K, "runs": {}, "B1": {}, "B2": {}, "B3": {}}
    runs = {}
    for v in VARIANTS:
        for rd in sorted(glob.glob(str(RES / v / "*/"))):
            rd = Path(rd)
            if (rd / "results.yaml").exists():
                d = load_results(rd)
                if "enhanced_clip_vitl14_hicem_discovered_concept_auc" in d:
                    runs[f"{v}/{rd.name}"] = (rd, d)
    # B1: пересечение сопоставленных понятий во всех прогонах
    out["runs"] = {r: str(rd) for r, (rd, d) in runs.items()}
    aucs = {r: matched_test_aucs(d) for r, (rd, d) in runs.items()}
    nonempty = [set(a) for a in aucs.values() if a]
    common = set.intersection(*nonempty) if nonempty else set()
    out["B1"]["n_common"] = len(common)
    fmt = lambda v: "—" if v is None else f"{v:.4f}"  # noqa: E731
    for r, a in aucs.items():
        out["B1"][r] = {"n_matched": len(a), "mean_all": float(np.mean(list(a.values()))) if a else None,
                        "mean_common": float(np.mean([a[s] for s in common])) if (a and common) else None}
        print(f"B1 {r:30s} сопоставлено {len(a):4d}; ROC-AUC по своим {fmt(out['B1'][r]['mean_all'])}; "
              f"по {len(common)} общим {fmt(out['B1'][r]['mean_common'])}", flush=True)

    # B2: пол «родитель как оценка» на тесте
    def parent_floor(j):
        if j not in parent:
            return 0.5
        s = cte[:, parent[j]].astype(float)
        y = test_gt[:, j]
        P, N = y.sum(), len(y) - y.sum()
        if P == 0 or N == 0:
            return float("nan")
        tpr, fpr = (s * y).sum() / P, (s * (1 - y)).sum() / N
        return float(0.5 + 0.5 * (tpr - fpr))

    for r, a in aucs.items():
        if not a:
            continue
        margins = [a[s] - parent_floor(names.index(s)) for s in a if s in names]
        floors = [parent_floor(names.index(s)) for s in a if s in names]
        out["B2"][r] = {"floor_mean": float(np.nanmean(floors)), "margin_mean": float(np.nanmean(margins)),
                        "share_margin_lt_0.05": float(np.nanmean(np.array(margins) < 0.05))}
        print(f"B2 {r:30s} пол родителя {out['B2'][r]['floor_mean']:.4f}; запас HiCEM {out['B2'][r]['margin_mean']:+.4f}; "
              f"доля понятий с запасом < 0.05: {out['B2'][r]['share_margin_lt_0.05']:.2f}", flush=True)

    # B3: нулевые модели
    rng = np.random.RandomState(0)
    for r, (rd, d) in runs.items():
        X = np.load(rd / "discovered_concepts.npz")["discovered_concept_labels"].astype(np.float32)
        colsum = X.sum(0)
        n_sub = d["n_discovered_sub_concepts"]
        cnt, auc, best_real, owned = best_per_concept(X, colsum, bank, n_sub, ds.sub_concept_map)
        res = {"real": {"count": cnt, "auc": auc}}
        for kind in ("within", "parent", "image"):
            counts, aucs_n, ge = [], [], np.zeros(bank.shape[1])
            for _ in range(K):
                if kind == "within":
                    bn = bank_cls.copy()
                    for j in range(bank.shape[1]):
                        pool = np.where(coarse_cls[:, parent[j]] == 1)[0] if j in parent else np.arange(n_cls)
                        bn[pool, j] = bank_cls[rng.permutation(pool), j]
                    Y = bn[ytr]
                elif kind == "parent":
                    bn = bank_cls.copy()
                    tops = sorted(set(parent.values()))
                    for top in tops:
                        pool = np.where(coarse_cls[:, top] == 1)[0]
                        perm = rng.permutation(pool)
                        cols = [j for j in range(bank.shape[1]) if parent.get(j) == top]
                        bn[np.ix_(pool, cols)] = bank_cls[np.ix_(perm, cols)]
                    free = [j for j in range(bank.shape[1]) if j not in parent]
                    perm = rng.permutation(n_cls)
                    bn[:, free] = bank_cls[perm][:, free]
                    Y = bn[ytr]
                else:
                    Y = bank.copy()
                    for j in range(bank.shape[1]):
                        pool = np.where(ctr[:, parent[j]] == 1)[0] if j in parent else np.arange(len(ytr))
                        Y[pool, j] = bank[rng.permutation(pool), j]
                c_n, a_n, best_n, _ = best_per_concept(X, colsum, Y, n_sub, ds.sub_concept_map)
                counts.append(c_n), aucs_n.append(a_n)
                ge += np.nan_to_num(best_n, nan=-1) >= np.nan_to_num(best_real, nan=2)
            p = (1 + ge[list(owned)]) / (1 + K)
            res[kind] = {"count_mean": float(np.mean(counts)), "count_sd": float(np.std(counts)),
                         "auc_mean": float(np.nanmean(aucs_n)), "excess": float(cnt - np.mean(counts)),
                         "n_sig_bh_q0.1": bh(p) if len(p) else 0}
            print(f"B3 {r:30s} {kind:6s}: настоящее {cnt}, нуль {np.mean(counts):.1f} ± {np.std(counts):.1f} "
                  f"(ROC-AUC {np.nanmean(aucs_n):.4f}); сверх нуля {cnt - np.mean(counts):+.1f}; "
                  f"значимых при BH q = 0.1: {res[kind]['n_sig_bh_q0.1']} из {len(p)}", flush=True)
        out["B3"][r] = res
        json.dump(out, open(RES / OUT_NAME, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
