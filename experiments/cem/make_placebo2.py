#!/usr/bin/env python
"""Пустышка-2 для CEM: копии train/val/test.pkl Koh et al., где каждый вид получает ЦЕЛИКОМ вектор 112 атрибутов
другого вида: одна перестановка видов pi на все атрибуты сразу, P[c, :] = L[pi(c), :] (pi без неподвижных точек,
чтобы ни один вид не сохранил свои ответы). В отличие от пустышки-1 (своя перестановка на каждый атрибут)
сохраняются частоты, связи между атрибутами, число «да» в строках и различимость видов по любому набору атрибутов;
разрывается только связь ответов с внешностью. Зерно 3000 + разбиение.
Результат: ~/cem_repro/results/cub_subsample_0.25/placebo2_data/fold_<k>/{train,val,test}.pkl
"""

import os
import pickle
import sys

import numpy as np

REPO = os.path.expanduser("~/cem_repro/cem")
BASE_DIR = os.path.join(REPO, "cem", "data", "CUB200", "class_attr_data_10")
RES = os.path.expanduser("~/cem_repro/results/cub_subsample_0.25")
OUT = os.path.join(RES, "placebo2_data")

data = {s: pickle.load(open(os.path.join(BASE_DIR, f"{s}.pkl"), "rb")) for s in ("train", "val", "test")}
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
    rng = np.random.RandomState(3000 + fold)
    while True:  # перестановка без неподвижных точек
        pi = rng.permutation(200)
        if not np.any(pi == np.arange(200)):
            break
    P = L[pi]  # P[c, :] = L[pi(c), :]
    d_out = os.path.join(OUT, f"fold_{fold}")
    os.makedirs(d_out, exist_ok=True)
    for s, rows in data.items():
        new = []
        for d in rows:
            e = dict(d)
            e["attribute_label"] = P[d["class_label"]].tolist()
            new.append(e)
        pickle.dump(new, open(os.path.join(d_out, f"{s}.pkl"), "wb"))
    sel = np.load(os.path.join(RES, f"selected_concepts_sampling_0.25_fold_{fold}.npy"))
    n_real = len({tuple(r) for r in L[:, sel]})
    n_pl = len({tuple(r) for r in P[:, sel]})
    print(f"fold {fold}: совпадение с настоящими метками {float((P == L).mean()):.3f}; частоты сохранены: "
          f"{np.allclose(P.mean(0), L.mean(0))}; разных наборов из 28: настоящие {n_real}, пустышка-2 {n_pl}")
