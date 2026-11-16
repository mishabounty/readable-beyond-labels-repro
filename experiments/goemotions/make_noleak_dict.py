#!/usr/bin/env python
"""Словарь n-грамм без утечки концептов людей: убираются n-граммы с |phi| > порога (на fit) хотя бы с одним
концептом словаря --eval-dict. Оставшиеся идут в прежнем порядке. Результат: data/<ds>/concepts_ngram_no<eval>.{npz,json} (или --name).
(Для CEBaB есть make_noaspect_dict.py, который дополнительно убирает слова лексикона аспектов.)"""

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", required=True)
ap.add_argument("--eval-dict", required=True)
ap.add_argument("--thr", type=float, default=0.05)
ap.add_argument("--name", help="имя словаря вместо ngram_no<eval> (например, ngram_phi0.1)")
args = ap.parse_args()
d = Path(__file__).resolve().parent / "data" / args.dataset
names = json.loads((d / "concepts_ngram.json").read_text())
N = sp.load_npz(d / "concepts_ngram.npz").tocsr()
A = sp.load_npz(d / f"concepts_{args.eval_dict}.npz").tocsr()
fit = np.load(d / "split.npz")["fit"]
X, Y = N[fit].toarray().astype(np.float32), A[fit].toarray().astype(np.float32)
X = (X - X.mean(0)) / (X.std(0) + 1e-9)
Y = (Y - Y.mean(0)) / (Y.std(0) + 1e-9)
phi = np.abs(X.T @ Y / len(X)).max(1)
keep = [i for i in range(len(names)) if phi[i] <= args.thr]
print(f"оставлено {len(keep)} из {len(names)}; убраны, например:",
      [names[i] for i in np.argsort(-phi)[:15]])
name = args.name or f"ngram_no{args.eval_dict}"
sp.save_npz(d / f"concepts_{name}.npz", N[:, keep].tocsr())
(d / f"concepts_{name}.json").write_text(json.dumps([names[i] for i in keep], ensure_ascii=False))
