#!/usr/bin/env python
"""Делит словарь концептов людей пополам (случайно, фиксированный seed): <имя>A -- для обучения этапа 1,
<имя>B -- отложенные, для проверки читаемости. Результат: data/<ds>/concepts_<имя>{A,B}.{npz,json}, где <имя> --
--name (по умолчанию --dict). --stratify: половина каждой группы (для GoEmotions -- группы тональности Demszky et al.,
2020) уходит в A, половина в B; при нечётном размере группы лишний концепт попадает то в A, то в B по очереди."""

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp

GROUPS = {"goemotions": {  # тональность эмоций GoEmotions (Demszky et al., 2020)
    "positive": ["admiration", "amusement", "approval", "caring", "desire", "excitement", "gratitude", "joy", "love",
                 "optimism", "pride", "relief"],
    "negative": ["anger", "annoyance", "disappointment", "disapproval", "disgust", "embarrassment", "fear", "grief",
                 "nervousness", "remorse", "sadness"],
    "ambiguous": ["confusion", "curiosity", "realization", "surprise"]}}

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", required=True)
ap.add_argument("--dict", required=True)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--name", help="имя пары словарей вместо --dict (например, emotion_r1)")
ap.add_argument("--stratify", action="store_true")
args = ap.parse_args()
d = Path(__file__).resolve().parent / "data" / args.dataset
names = json.loads((d / f"concepts_{args.dict}.json").read_text())
M = sp.load_npz(d / f"concepts_{args.dict}.npz").tocsr()
rng = np.random.default_rng(args.seed)
if args.stratify:
    a_idx, extra_to_a = [], bool(args.seed % 2)
    for g, members in GROUPS[args.dataset].items():
        idx = rng.permutation([names.index(n) for n in members])
        half = len(idx) // 2 + (len(idx) % 2 if extra_to_a else 0)
        extra_to_a ^= bool(len(idx) % 2)
        a_idx += list(idx[:half])
    parts = {"A": sorted(a_idx), "B": sorted(set(range(len(names))) - set(a_idx))}
else:
    perm = rng.permutation(len(names))
    parts = {"A": sorted(perm[: len(names) // 2]), "B": sorted(perm[len(names) // 2:])}
prefix = args.name or args.dict
for part, idx in parts.items():
    sp.save_npz(d / f"concepts_{prefix}{part}.npz", M[:, idx].tocsr())
    (d / f"concepts_{prefix}{part}.json").write_text(json.dumps([names[i] for i in idx]))
    print(prefix, part, len(idx), [names[i] for i in idx])
