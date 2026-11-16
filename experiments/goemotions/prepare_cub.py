#!/usr/bin/env python
"""CUB-200-2011 в формате этой папки (как prepare_data.py для текстов).

Вход: распакованный архив data/cub/CUB_200_2011 (корень проекта). Признаки изображений считает embed_cub.py.
  labels.npy      -- вид птицы (200 классов);
  split.npz       -- fit / val (15% официального train, стратифицированно) / test (официальный test);
  concepts_cub{A,B} -- атрибуты на уровне изображения (is_present, 312 атрибутов, оставлены встречающиеся
                     хотя бы у 1% изображений); группы атрибутов («has_wing_color::...») случайно делятся пополам:
                     A -- для обучения этапа 1, B -- отложенные (группы не пересекаются);
  meta.json, texts.json (пути к изображениям, для совместимости).
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy.sparse as sp

HERE = Path(__file__).resolve().parent
SRC = HERE.parents[1] / "data" / "cub" / "CUB_200_2011"
OUT = HERE / "data" / "cub"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    paths = [l.split()[1] for l in (SRC / "images.txt").read_text().splitlines()]
    y = np.array([int(l.split()[1]) - 1 for l in (SRC / "image_class_labels.txt").read_text().splitlines()])
    is_train = np.array([int(l.split()[1]) for l in (SRC / "train_test_split.txt").read_text().splitlines()]) == 1
    names_file = SRC / "attributes" / "attributes.txt"
    if not names_file.exists():
        names_file = SRC.parent / "attributes.txt"
    names = [l.split(" ", 1)[1] for l in names_file.read_text().splitlines()]
    A = np.zeros((len(paths), len(names)), dtype=np.int8)
    for l in (SRC / "attributes" / "image_attribute_labels.txt").read_text().splitlines():
        parts = l.split()
        if len(parts) >= 3 and parts[2] == "1":
            A[int(parts[0]) - 1, int(parts[1]) - 1] = 1
    keep = np.where(A.mean(0) >= 0.01)[0]
    groups = defaultdict(list)
    for j in keep:
        groups[names[j].split("::")[0]].append(j)
    order = sorted(groups)
    perm = np.random.default_rng(0).permutation(len(order))
    half = {order[i] for i in perm[: len(order) // 2]}
    parts = {"A": [j for g in order if g in half for j in groups[g]],
             "B": [j for g in order if g not in half for j in groups[g]]}
    for part, cols in parts.items():
        sp.save_npz(OUT / f"concepts_cub{part}.npz", sp.csr_matrix(A[:, cols]))
        (OUT / f"concepts_cub{part}.json").write_text(json.dumps([names[j] for j in cols]))
        print(part, len(cols), "атрибутов,", len({names[j].split("::")[0] for j in cols}), "групп")
    rng = np.random.default_rng(0)
    train_idx, test_idx = np.where(is_train)[0], np.where(~is_train)[0]
    val = np.concatenate([rng.permutation(train_idx[y[train_idx] == c])[: max(1, int(0.15 * (y[train_idx] == c).sum()))]
                          for c in range(200)])
    fit = np.setdiff1d(train_idx, val)
    np.savez(OUT / "split.npz", fit=fit, val=np.sort(val), test=test_idx)
    np.save(OUT / "labels.npy", y)
    (OUT / "texts.json").write_text(json.dumps(paths))
    (OUT / "meta.json").write_text(json.dumps({"dataset": "cub", "num_classes": 200, "fit": len(fit), "val": len(val),
                                               "test": len(test_idx), "attributes_kept": len(keep)}, indent=1))
    print(f"fit {len(fit)} val {len(val)} test {len(test_idx)}")


if __name__ == "__main__":
    main()
