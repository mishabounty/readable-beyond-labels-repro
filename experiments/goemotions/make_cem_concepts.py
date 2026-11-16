#!/usr/bin/env python
"""Атрибуты CUB для проверки опыта CEM §5.3 (Espinosa Zarlenga et al., 2022): обучение на 28 случайных атрибутах,
чтение остальных.

Атрибуты вида по протоколу CBM (Koh et al., 2020): отметки «не видно» (certainty 1) у отсутствующих атрибутов
пропускаются, атрибут вида = большинство голосов изображений вида из fit (ничья -> 1); оставлены атрибуты,
присутствующие хотя бы у 10 видов (у Koh 112, здесь получается чуть меньше: другое разбиение fit). Каждое
изображение получает вектор своего вида.
  cemA    -- 28 случайных атрибутов (seed 0), как в CEM §5.3;
  cemB    -- остальные атрибуты (только для чтения);
  cemArnd -- пустышка той же формы: для каждого атрибута A случайный набор видов того же размера
             (функция вида, как настоящие атрибуты, но без их смысла).
Результат: data/cub/concepts_cem{A,B,Arnd}.{npz,json}.
"""

import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp

HERE = Path(__file__).resolve().parent
d = HERE / "data" / "cub"
RAW = HERE.parents[1] / "data" / "cub" / "CUB_200_2011"
y = np.load(d / "labels.npy")
fit = set(np.load(d / "split.npz")["fit"].tolist())
names = [l.split(" ", 1)[1].strip() for l in (RAW / "attributes" / "attributes.txt").read_text().splitlines()] \
    if (RAW / "attributes" / "attributes.txt").exists() else \
    [l.split(" ", 1)[1].strip() for l in (RAW.parent / "attributes.txt").read_text().splitlines()]
n_att = len(names)
cnt = np.zeros((200, n_att, 2))
for l in (RAW / "attributes" / "image_attribute_labels.txt").read_text().splitlines():
    f = l.split()
    i, j, a, cert = int(f[0]) - 1, int(f[1]) - 1, int(f[2]), int(f[3])
    if i not in fit or (a == 0 and cert == 1):
        continue
    cnt[y[i], j, a] += 1
cls = (cnt[..., 1] >= cnt[..., 0]).astype(np.int8)
keep = np.where(cls.sum(0) >= 10)[0]
rng = np.random.default_rng(0)
perm = rng.permutation(len(keep))
a_idx, b_idx = np.sort(keep[perm[:28]]), np.sort(keep[perm[28:]])
# пустышка: у атрибута A столько же положительных видов, но виды выбраны случайно
rnd = np.zeros((200, 28), dtype=np.int8)
for k, j in enumerate(a_idx):
    rnd[rng.choice(200, int(cls[:, j].sum()), replace=False), k] = 1
for tag, M, nm in (("cemA", cls[:, a_idx], [names[j] for j in a_idx]),
                   ("cemB", cls[:, b_idx], [names[j] for j in b_idx]),
                   ("cemArnd", rnd, [f"random_{k}" for k in range(28)])):
    sp.save_npz(d / f"concepts_{tag}.npz", sp.csr_matrix(M[y]))
    (d / f"concepts_{tag}.json").write_text(json.dumps(nm, ensure_ascii=False))
print(f"атрибутов вида: {len(keep)}; A = {len(a_idx)}, B = {len(b_idx)}; "
      f"доля положительных в B на изображение {cls[:, b_idx][y].mean():.3f}")
