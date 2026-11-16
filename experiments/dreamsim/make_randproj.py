#!/usr/bin/env python
"""Годная пустышка для DreamSim — «случайная проекция» (план experiments/PLAN_2026-11-11_amendments.md, П1): ответ тройки —
какая из двух искажённых картинок ближе к эталону по косинусному расстоянию в случайной проекции (768 → 64, гауссова
матрица, зерно 9100 + k) cls-признаков исходной DINO ViT-B/16. Согласована с картинками и выучивается, суждений людей не
несёт (но наследует сходство исходной DINO — совпадение с людьми сообщается как свойство). Меняются train и val (votes ≥ 6),
test — настоящий. Результат: dataset/nights_rp_s<k>/ (data.csv + ссылки на картинки), кэш признаков dataset/dino_cls.npz.
Запуск (из ~/dreamsim_repro/code, на GPU): python ../make_randproj.py"""

import os
import sys

import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, os.getcwd())
from dreamsim import PerceptualModel  # noqa: E402
from util.utils import get_preprocess_fn  # noqa: E402
from torchvision import transforms  # noqa: E402

H = os.path.expanduser("~/dreamsim_repro")
D = os.path.join(H, "dataset")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
src = pd.read_csv(os.path.join(D, "nights", "data.csv"))
use = (src["votes"] >= 6) & src["split"].isin(["train", "val"])
paths = sorted(set(src.loc[use, ["ref_path", "left_path", "right_path"]].values.ravel()))
cache = os.path.join(D, "dino_cls.npz")
if os.path.exists(cache):
    z = np.load(cache, allow_pickle=True)
    emb = dict(zip(z["paths"], z["emb"]))
else:
    model = PerceptualModel(model_type="dino_vitb16", feat_type="cls", stride="16", baseline=True,
                            load_dir=os.path.join(H, "models"), device=DEV).to(DEV).eval()
    pre = get_preprocess_fn("DEFAULT", 224, transforms.InterpolationMode.BICUBIC)
    vecs = []
    with torch.no_grad():
        for i in range(0, len(paths), 64):
            x = torch.stack([pre(Image.open(os.path.join(D, "nights", p))) for p in paths[i:i + 64]]).to(DEV)
            vecs.append(model.embed(x).float().cpu().numpy().reshape(len(x), -1))
            if i % 6400 == 0:
                print(f"признаки {i}/{len(paths)}", flush=True)
    E = np.concatenate(vecs)
    np.savez(cache, paths=np.array(paths), emb=E)
    emb = dict(zip(paths, E))
print("картинок", len(emb), "размерность", len(next(iter(emb.values()))), flush=True)


def cosd(a, b):
    return 1 - (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)


for k in range(5):
    R = np.random.RandomState(9100 + k).standard_normal((768, 64)).astype(np.float32)
    df = src.copy()
    idx = np.where(use)[0]
    ref = np.stack([emb[p] for p in df.loc[idx, "ref_path"]]) @ R
    left = np.stack([emb[p] for p in df.loc[idx, "left_path"]]) @ R
    right = np.stack([emb[p] for p in df.loc[idx, "right_path"]]) @ R
    rv = (cosd(ref, right) < cosd(ref, left)).astype(int)
    df.loc[idx, "right_vote"] = rv
    df.loc[idx, "left_vote"] = 1 - rv
    d = os.path.join(D, f"nights_rp_s{k}")
    os.makedirs(d, exist_ok=True)
    for sub in ("ref", "distort"):
        link = os.path.join(d, sub)
        if not os.path.islink(link):
            os.symlink(os.path.join(D, "nights", sub), link)
    df.to_csv(os.path.join(d, "data.csv"), index=False)
    for split in ("train", "val"):
        m = use & (df["split"] == split)
        print(f"nights_rp_s{k} {split}: троек {int(m.sum())}, доля «правая ближе» {df.loc[m, 'right_vote'].mean():.3f}, "
              f"совпадение с людьми {float((df.loc[m, 'right_vote'].values == src.loc[m, 'right_vote'].values).mean()):.3f}",
              flush=True)
print("RANDPROJ_DONE", flush=True)
