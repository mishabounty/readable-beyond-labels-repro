#!/usr/bin/env python
"""Условная информативность концептов этапа 1 (A) об отложенных концептах (B) при известной метке задачи y.

Для каждого отложенного концепта b -- логистическая регрессия на fit по признакам [one-hot(y)] и [one-hot(y), A]
(A -- бинарные метки концептов этапа 1, C по val AP из сетки); на тесте:
  macro-AP по y и по [y, A] и их разность (сколько A добавляют к тому, что уже даёт метка задачи);
  AUC внутри класса задачи для [y, A] (для одного y она равна 0.5): различают ли метки A примеры одного класса
  с концептом b и без него.
Это верхняя оценка того, что могла бы передать представлению супервизия A сверх задачи.
Результат дописывается в results/informativeness.jsonl (ключ -- dataset, train_dict, eval_dict, k).
"""

import argparse
import json
import os
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import scipy.sparse as sp
import torch
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

import train as T
from collapse_check import within_class_auc

HERE = Path(__file__).resolve().parent
CS = [1e-3, 1e-2, 1e-1, 1.0, 10.0]


def _fit(Xtr, ytr, Xva, yva, Xte):
    if ytr.sum() == 0:
        return np.zeros(Xte.shape[0])
    best = None
    for C in CS:
        clf = LogisticRegression(C=C, max_iter=3000).fit(Xtr, ytr)
        ap = average_precision_score(yva, clf.decision_function(Xva)) if yva.sum() > 0 else 0.0
        if best is None or ap > best[0]:
            best = (ap, clf)
    return best[1].decision_function(Xte)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("specs", nargs="+", help="dataset:train_dict:eval_dict:k")
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args()
    out = HERE / "results" / "informativeness.jsonl"
    done = {(r["dataset"], r["train_dict"], r["eval_dict"], r["k"]) for r in map(json.loads, open(out))} \
        if out.exists() else set()
    for spec in args.specs:
        ds, td, ev, k = spec.split(":")
        k = int(k)
        if (ds, td, ev, k) in done:
            continue
        _, y, split, MA = T.load(ds, td, k, "mqa" if ds in ("cebab", "goemotions") else "inception", False, 0)
        MB = T.load(ds, ev, 0, "mqa" if ds in ("cebab", "goemotions") else "inception", False, 0)[3]
        n_cls = int(y.max()) + 1
        Yoh = sp.csr_matrix(np.eye(n_cls, dtype=np.float32)[y.numpy()])
        feats = {"y": Yoh, "yA": sp.hstack([Yoh, MA.astype(np.float32)]).tocsr()}
        fit, val, te = (split[s].numpy() for s in ("fit", "val", "test"))
        B = MB.toarray().astype(np.float32)
        row = {"dataset": ds, "train_dict": td, "eval_dict": ev, "k": k, "n_A": MA.shape[1], "n_B": MB.shape[1]}
        for name, X in feats.items():
            S = np.stack(Parallel(n_jobs=args.jobs)(delayed(_fit)(X[fit], B[fit, j], X[val], B[val, j], X[te])
                                                    for j in range(B.shape[1])), 1)
            m = T.concept_metrics(torch.tensor(S, dtype=torch.float32), torch.tensor(B[te]))
            row[f"macro_ap_{name}"] = m["macro_ap"]
            row[f"ap_{name}"] = m["ap"]
            row[f"auc_within_{name}"] = within_class_auc(torch.tensor(S), torch.tensor(B[te]), y[torch.as_tensor(te)])
        row["gain_macro_ap"] = row["macro_ap_yA"] - row["macro_ap_y"]
        with open(out, "a") as fh:
            fh.write(json.dumps(row) + "\n")
        print(json.dumps({k_: (round(v, 3) if isinstance(v, float) else v) for k_, v in row.items()}), flush=True)


if __name__ == "__main__":
    main()
