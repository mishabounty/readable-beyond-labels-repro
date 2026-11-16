#!/usr/bin/env python
"""Быстрая сводка пробы до сходимости по одному набору z: среднее по seed для каждого (метод, lambda, отбеливание).

python ridge_summary.py <набор> [префикс=probe_ridge_] [jsonl этапа 1]
  <набор>     -- имя из results/z/ (например finetune_goemotions_length_v2);
  префикс     -- probe_ridge_ (ridge, probe_ridge.py) или probe_ (логистическая регрессия Ньютоном, probe_eval.py);
  jsonl этапа 1 -- results/collapse_<...>.jsonl или results/finetune_<...>.jsonl: рядом печатаются прежняя проба
                 (old ap, aucw -- AdamW 30 эпох, НЕ использовать как читаемость), eff. rank, F1 задачи, AP концептов A.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parent / "results"
name = sys.argv[1]
pref = sys.argv[2] if len(sys.argv) > 2 else "probe_ridge_"
oldf = sys.argv[3] if len(sys.argv) > 3 else None
g = defaultdict(list)
for l in open(R / f"{pref}{name}.jsonl"):
    r = json.loads(l)
    g[(r["method"], r["lambda"], r["white"])].append(r)
old = defaultdict(list)
if oldf:
    for l in open(oldf):
        r = json.loads(l)
        old[("input" if r["method"] == "raw_x" else r["method"], r["lambda"])].append(r)


def mean(rows, key):
    v = [r[key] for r in rows if r.get(key) is not None]
    return np.mean(v) if v else np.nan


for k, v in sorted(g.items(), key=lambda kv: (kv[0][2], kv[0][0], kv[0][1])):
    s = f"{k[0]:9s} {k[1]:4g} white={str(k[2]):5s} n={len(v)} " + " ".join(
        f"{m}={mean(v, m):.3f}" for m in ["macro_ap", "ap", "auc_within", "macro_ap_200"])
    o = old.get(k[:2], [])
    if o:
        s += (f" | old ap={mean(o, 'ap_full'):.3f} aucw={mean(o, 'auc_within'):.3f} erank={mean(o, 'effective_rank'):.0f}"
              f" F1={mean(o, 'task_f1'):.3f} a_ap={mean(o, 'a_ap'):.3f}")
    print(s)
