#!/usr/bin/env python
"""Другие случайные разбиения групп атрибутов CUB пополам (как в prepare_cub.py, но seed 1..): A -- этап 1,
B -- отложенные. Результат: data/cub/concepts_cub_g<seed>{A,B}.{npz,json} и ссылки в data/cub4."""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy.sparse as sp

HERE = Path(__file__).resolve().parent
D = HERE / "data" / "cub"
names = json.loads((D / "concepts_cubA.json").read_text()) + json.loads((D / "concepts_cubB.json").read_text())
M = sp.hstack([sp.load_npz(D / "concepts_cubA.npz"), sp.load_npz(D / "concepts_cubB.npz")]).tocsc()
groups = defaultdict(list)
for j, n in enumerate(names):
    groups[n.split("::")[0]].append(j)
order = sorted(groups)
for seed in map(int, sys.argv[1:]):
    perm = np.random.default_rng(seed).permutation(len(order))
    half = {order[i] for i in perm[: len(order) // 2]}
    parts = {"A": [j for g in order if g in half for j in groups[g]],
             "B": [j for g in order if g not in half for j in groups[g]]}
    for part, cols in parts.items():
        stem = f"concepts_cub_g{seed}{part}"
        sp.save_npz(D / f"{stem}.npz", M[:, cols].tocsr())
        (D / f"{stem}.json").write_text(json.dumps([names[j] for j in cols]))
        for ext in ("npz", "json"):
            link = HERE / "data" / "cub4" / f"{stem}.{ext}"
            if not link.exists():
                link.symlink_to(f"../cub/{stem}.{ext}")
        print(seed, part, len(cols), "атрибутов,", len({names[j].split("::")[0] for j in cols}), "групп")
