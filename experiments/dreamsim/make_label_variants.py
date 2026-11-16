#!/usr/bin/env python
"""Наборы ответов для пустышек DreamSim (план experiments/PLAN_2026-11-10_dreamsim.md).

Из dataset/nights/data.csv (их таблица троек NIGHTS) делаются копии с другими ответами людей; картинки — ссылки на
dataset/nights/{ref,distort}. Меняются только тройки, которые их загрузчик берёт в обучение (votes ≥ 6), и только в
частях train и val; test остаётся с настоящими ответами.
  nights_shuf_s<k> (k = 0..4) — ответы (left_vote, right_vote) переставлены между тройками внутри train и внутри val,
                                зерно 9000 + k: доля ответов «правая ближе» та же, связь ответа с картинками рвётся;
  nights_anti                 — ответ каждой тройки train и val заменён на противоположный («наперекор людям»).
Запуск: python make_label_variants.py (из ~/dreamsim_repro).
"""

import os

import numpy as np
import pandas as pd

ROOT = os.path.expanduser("~/dreamsim_repro/dataset")
src = pd.read_csv(os.path.join(ROOT, "nights", "data.csv"))
use = src["votes"] >= 6


def write(name, df):
    d = os.path.join(ROOT, name)
    os.makedirs(d, exist_ok=True)
    for sub in ("ref", "distort"):
        link = os.path.join(d, sub)
        if not os.path.islink(link):
            os.symlink(os.path.join(ROOT, "nights", sub), link)
    df.to_csv(os.path.join(d, "data.csv"), index=False)
    for split in ("train", "val", "test"):
        m = use & (df["split"] == split)
        agree = float((df.loc[m, "right_vote"].values == src.loc[m, "right_vote"].values).mean())
        print(f"{name:14s} {split:5s}: троек {int(m.sum()):6d}, доля «правая ближе» {df.loc[m, 'right_vote'].mean():.3f} "
              f"(настоящие {src.loc[m, 'right_vote'].mean():.3f}), совпадение с людьми {agree:.3f}", flush=True)


assert set(src["right_vote"].unique()) <= {0, 1} and (src["left_vote"] + src["right_vote"] == 1).all(), \
    "ожидались бинарные ответы left_vote / right_vote"
for k in range(5):
    df = src.copy()
    rng = np.random.RandomState(9000 + k)
    for split in ("train", "val"):
        idx = np.where(use & (df["split"] == split))[0]
        perm = rng.permutation(idx)
        df.loc[idx, ["left_vote", "right_vote"]] = src.loc[perm, ["left_vote", "right_vote"]].values
    write(f"nights_shuf_s{k}", df)
df = src.copy()
for split in ("train", "val"):
    idx = np.where(use & (df["split"] == split))[0]
    df.loc[idx, "right_vote"] = 1 - src.loc[idx, "right_vote"]
    df.loc[idx, "left_vote"] = 1 - src.loc[idx, "left_vote"]
write("nights_anti", df)
