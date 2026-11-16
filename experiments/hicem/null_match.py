#!/usr/bin/env python
"""Нулевая калибровка сопоставления подпонятий с банком в HiCEM (их match_discovered_concepts_to_concept_bank).

Сопоставление выбирает для каждого понятия банка лучший из сотен–тысяч кандидатов по ROC-AUC (порог 0.7), поэтому
часть совпадений возможна случайно. Здесь то же сопоставление повторяется (векторно; для двоичных меток кандидата
ROC-AUC = (1 + TPR − FPR) / 2, ровно как sklearn) на настоящем банке — проверка, что воспроизводится их число
сопоставленных понятий, — и на K нулевых банках: метки каждого понятия банка перемешаны между видами внутри видов,
у которых родительское грубое понятие = 1 (вне их не меняются); для понятий без родителя — между всеми видами.
Кандидаты (discovered_concepts.npz) те же. Итог: сколько понятий «находится» и с каким ROC-AUC, когда банк случаен.
Запуск из ~/hicem_repro/code/experiments: python null_match.py <config.yaml> <run_dir> [<run_dir> ...] --k 20
"""

import argparse
from pathlib import Path

import numpy as np
import yaml

from experiment_utils import load_config, load_datasets


def match(X, colsum, Y, n_sub, sub_map, thr=0.7):
    """Их правило: блок кандидатов родителя (или все), лучший по ROC-AUC (первый максимум), > thr, и кандидат
    переходит к понятию банка с большим ROC-AUC. Возвращает (число сопоставленных, их средний ROC-AUC)."""
    n, m = X.shape
    starts = np.concatenate([[0], np.cumsum(n_sub)]).astype(int)
    P = Y.sum(0)
    N = n - P
    TP = Y.T @ X  # B × m
    FP = colsum[None, :] - TP
    with np.errstate(divide="ignore", invalid="ignore"):
        AUC = 0.5 + 0.5 * (TP / P[:, None] - FP / N[:, None])
    best_auc = np.zeros(m)
    owner = np.full(m, -1)
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
        if a[k] > thr and a[k] > best_auc[lo + k]:
            best_auc[lo + k] = a[k]
            owner[lo + k] = j
    ok = owner >= 0
    return int(ok.sum()), float(best_auc[ok].mean()) if ok.any() else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--k", type=int, default=20)
    args = ap.parse_args()
    ds = load_datasets(load_config(args.config))[0]
    y = ds.train_y.numpy().astype(int)
    c = ds.train_c.numpy()
    bank = ds.concept_bank.astype(np.float32)
    n_cls = ds.n_tasks
    first = np.array([np.where(y == k)[0][0] for k in range(n_cls)])
    bank_cls, coarse_cls = bank[first], c[first]  # банк и грубые понятия заданы на уровне вида
    assert np.array_equal(bank, bank_cls[y]) and np.array_equal(c, coarse_cls[y])
    parent = {}
    for top, members in enumerate(ds.sub_concept_map):
        for j in members:
            parent.setdefault(j, top)
    for rd in args.run_dirs:
        rd = Path(rd)
        X = np.load(rd / "discovered_concepts.npz")["discovered_concept_labels"].astype(np.float32)
        res = {}
        for doc in yaml.safe_load_all(open(rd / "results.yaml")):
            if isinstance(doc, dict):
                res.update(doc)
        n_sub = res["n_discovered_sub_concepts"]
        logged = int((np.array(res["discovered_concept_roc_aucs"], float) > 0).sum())
        colsum = X.sum(0)
        cnt, auc = match(X, colsum, bank, n_sub, ds.sub_concept_map)
        print(f"== {rd}: кандидатов {X.shape[1]}; настоящий банк: сопоставлено {cnt} (в их журнале {logged}), "
              f"средний ROC-AUC {auc:.4f}", flush=True)
        rng = np.random.RandomState(0)
        null = []
        for _ in range(args.k):
            bnull = bank_cls.copy()
            for j in range(bank.shape[1]):
                pool = np.where(coarse_cls[:, parent[j]] == 1)[0] if j in parent else np.arange(n_cls)
                bnull[pool, j] = bank_cls[rng.permutation(pool), j]
            null.append(match(X, colsum, bnull[y], n_sub, ds.sub_concept_map))
        cn = np.array([a for a, _ in null])
        an = np.array([b for _, b in null])
        print(f"   нулевой банк (K={args.k}): сопоставлено {cn.mean():.1f} ± {cn.std():.1f} "
              f"[{cn.min()}; {cn.max()}], средний ROC-AUC {np.nanmean(an):.4f}; "
              f"доля нулевых ≥ настоящего: {(cn >= cnt).mean():.2f}", flush=True)


if __name__ == "__main__":
    main()
