#!/usr/bin/env python
"""Пустышка «случайный код» для CEM (план experiments/PLAN_2026-11-10_checks.md, Г2): каждому виду — случайный двоичный
код, «да» с вероятностью 0.5 независимо по каждому из 112 атрибутов (учится модель на 28 выбранных); все 200 кодов
по 28 выбранным атрибутам разные (совпавшие переразыгрываются). Коды не зависят ни от картинок, ни от настоящих
меток. Зерно 5000 + повтор. Метки уровня вида, как у настоящих.
Результат: ~/cem_repro/results/cub_subsample_0.25/placebo4_data/fold_<k>/{train,val,test}.pkl и
placebo4_properties.json (доли, различимость видов, средний |φ| — рядом с теми же мерами настоящих меток).
"""

import json
import os
import pickle
import sys

import numpy as np

REPO = os.path.expanduser("~/cem_repro/cem")
BASE_DIR = os.path.join(REPO, "cem", "data", "CUB200", "class_attr_data_10")
RES = os.path.expanduser("~/cem_repro/results/cub_subsample_0.25")
OUT = os.path.join(RES, "placebo4_data")


def mean_abs_phi(M):
    M = M.astype(float)
    M = M[:, M.std(0) > 0]
    C = np.corrcoef(M.T)
    return float(np.abs(C[np.triu_indices(len(C), 1)]).mean())


data = {s: pickle.load(open(os.path.join(BASE_DIR, f"{s}.pkl"), "rb")) for s in ("train", "val", "test")}
table = {}
for s in data:
    for d in data[s]:
        lab = np.array(d["attribute_label"], dtype=np.int64)
        c = d["class_label"]
        if c in table and not np.array_equal(table[c], lab):
            sys.exit(f"атрибуты вида {c} различаются между изображениями — пустышка по виду неприменима")
        table[c] = lab
L = np.stack([table[c] for c in range(200)])  # 200 x 112, настоящие
props = {}
for fold in range(5):
    sel = np.load(os.path.join(RES, f"selected_concepts_sampling_0.25_fold_{fold}.npy"))
    rng = np.random.RandomState(5000 + fold)
    P = (rng.rand(200, L.shape[1]) < 0.5).astype(np.int64)
    while True:  # все коды по 28 выбранным атрибутам разные
        keys = [tuple(r) for r in P[:, sel]]
        dup = [i for i, k in enumerate(keys) if keys.index(k) != i]
        if not dup:
            break
        for i in dup:
            P[i, sel] = (rng.rand(len(sel)) < 0.5).astype(np.int64)
    d_out = os.path.join(OUT, f"fold_{fold}")
    os.makedirs(d_out, exist_ok=True)
    for s, rows in data.items():
        new = []
        for d in rows:
            e = dict(d)
            e["attribute_label"] = P[d["class_label"]].tolist()
            new.append(e)
        pickle.dump(new, open(os.path.join(d_out, f"{s}.pkl"), "wb"))
    props[fold] = {"distinct_real": len({tuple(r) for r in L[:, sel]}), "distinct_placebo4": len({tuple(r) for r in P[:, sel]}),
                   "rate_real": float(L[:, sel].mean()), "rate_placebo4": float(P[:, sel].mean()),
                   "phi_real": mean_abs_phi(L[:, sel]), "phi_placebo4": mean_abs_phi(P[:, sel]),
                   "agree_with_real": float((P[:, sel] == L[:, sel]).mean())}
    print(f"fold {fold}: {props[fold]}", flush=True)
json.dump(props, open(os.path.join(RES, "placebo4_properties.json"), "w"), indent=1)
