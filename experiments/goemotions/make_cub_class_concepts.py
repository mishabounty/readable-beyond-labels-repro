#!/usr/bin/env python
"""Атрибуты CUB по протоколу CBM (Koh et al., 2020): для каждого вида атрибут присутствует, если он отмечен хотя бы
у половины изображений вида (по fit); оставлены атрибуты, присутствующие хотя бы у 10 видов; каждое изображение
получает вектор своего вида. Делается для части A (обучение этапа 1); B остаётся на уровне изображений.
Результат: data/cub/concepts_cubAcls.{npz,json} и ссылки в data/cub4."""

import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp

d = Path(__file__).resolve().parent / "data" / "cub"
A = sp.load_npz(d / "concepts_cubA.npz").toarray()
names = json.loads((d / "concepts_cubA.json").read_text())
y = np.load(d / "labels.npy")
fit = np.load(d / "split.npz")["fit"]
cls = np.stack([A[fit][y[fit] == c].mean(0) >= 0.5 for c in range(200)]).astype(np.int8)
keep = np.where(cls.sum(0) >= 10)[0]
M = cls[:, keep][y]
sp.save_npz(d / "concepts_cubAcls.npz", sp.csr_matrix(M))
(d / "concepts_cubAcls.json").write_text(json.dumps([names[j] for j in keep]))
for f in ("concepts_cubAcls.npz", "concepts_cubAcls.json"):
    link = d.parent / "cub4" / f
    if not link.exists():
        link.symlink_to(Path("..") / "cub" / f)
print(f"атрибутов вида: {len(keep)} из {len(names)}; в среднем {M.sum(1).mean():.1f} на изображение")
