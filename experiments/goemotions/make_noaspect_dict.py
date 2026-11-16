#!/usr/bin/env python
"""Словарь n-грамм CEBaB без утечки аспектов: убираются n-граммы, связанные с любым аспектом (|phi| > порога
на fit) или содержащие слово из лексикона аспектов. Оставшиеся идут в прежнем порядке (по частоте).
Результат: data/cebab/concepts_ngram_noaspect.{npz,json}."""

import json
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

d = Path(__file__).resolve().parent / "data" / "cebab"
thr = float(sys.argv[1]) if len(sys.argv) > 1 else 0.05
LEX = ["food", "service", "ambiance", "ambience", "atmosphere", "noise", "noisy", "loud", "quiet", "waiter",
       "waitress", "staff", "server", "decor", "music", "delicious", "tasty", "dish", "meal", "hear", "attentive",
       "rude", "slow", "bland", "view"]
names = json.loads((d / "concepts_ngram.json").read_text())
N = sp.load_npz(d / "concepts_ngram.npz").tocsr()
A = sp.load_npz(d / "concepts_aspect.npz").tocsr()
fit = np.load(d / "split.npz")["fit"]
X, Y = N[fit].toarray().astype(np.float32), A[fit].toarray().astype(np.float32)
X = (X - X.mean(0)) / (X.std(0) + 1e-9)
Y = (Y - Y.mean(0)) / (Y.std(0) + 1e-9)
phi = np.abs(X.T @ Y / len(X)).max(1)
keep = [i for i, n in enumerate(names) if phi[i] <= thr and not any(w in n for w in LEX)]
print(f"оставлено {len(keep)} из {len(names)}; первые 30:", [names[i] for i in keep[:30]])
sp.save_npz(d / "concepts_ngram_noaspect.npz", N[:, keep].tocsr())
(d / "concepts_ngram_noaspect.json").write_text(json.dumps([names[i] for i in keep], ensure_ascii=False))
