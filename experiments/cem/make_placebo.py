#!/usr/bin/env python
"""Пустышка для CEM: копии train/val/test.pkl Koh et al., где метки каждого из 112 понятий переставлены между
видами (у понятия j своя случайная перестановка видов pi_j; изображение вида c получает метку вида pi_j(c)).
Частота каждого понятия на уровне видов та же, метки по-прежнему функция вида, но смысла (связи с изображением)
нет. При обучении используются только 28 выбранных понятий; проба читает настоящие метки из исходных файлов.
Результат: ~/cem_repro/results/cub_subsample_0.25/placebo_data/fold_<k>/{train,val,test}.pkl
"""

import os
import pickle
import sys

import numpy as np

REPO = os.path.expanduser("~/cem_repro/cem")
BASE_DIR = os.path.join(REPO, "cem", "data", "CUB200", "class_attr_data_10")
OUT = os.path.expanduser("~/cem_repro/results/cub_subsample_0.25/placebo_data")

data = {s: pickle.load(open(os.path.join(BASE_DIR, f"{s}.pkl"), "rb")) for s in ("train", "val", "test")}
# таблица атрибутов вида (у Koh атрибуты заданы на уровне вида: у всех изображений вида одинаковы)
table = {}
for s in data:
    for d in data[s]:
        lab = np.array(d["attribute_label"], dtype=np.int64)
        c = d["class_label"]
        if c in table and not np.array_equal(table[c], lab):
            sys.exit(f"атрибуты вида {c} различаются между изображениями -- пустышка по виду неприменима")
        table[c] = lab
L = np.stack([table[c] for c in range(200)])  # 200 x 112
for fold in range(5):
    rng = np.random.RandomState(2000 + fold)
    perms = np.stack([rng.permutation(200) for _ in range(L.shape[1])], 1)  # 200 x 112: pi_j(c)
    P = L[perms, np.arange(L.shape[1])]  # P[c, j] = L[pi_j(c), j]
    d_out = os.path.join(OUT, f"fold_{fold}")
    os.makedirs(d_out, exist_ok=True)
    for s, rows in data.items():
        new = []
        for d in rows:
            e = dict(d)
            e["attribute_label"] = P[d["class_label"]].tolist()
            new.append(e)
        pickle.dump(new, open(os.path.join(d_out, f"{s}.pkl"), "wb"))
    same = float((P == L).mean())
    print(f"fold {fold}: совпадение пустышки с настоящими метками {same:.3f}; "
          f"частоты понятий сохранены: {np.allclose(P.mean(0), L.mean(0))}")
