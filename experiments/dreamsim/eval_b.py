#!/usr/bin/env python
"""Оценка моделей DreamSim (план experiments/PLAN_2026-11-10_dreamsim.md): NIGHTS test (2AFC, настоящие ответы; область A)
и проверка на данных, не входивших в обучение (B): TID2013 и KADID-10k — SROCC и PLCC между −расстоянием модели
(эталон ↔ искажённое) и оценкой людей (MOS / DMOS: больше — лучше).
Модели: base (исходная DINO ViT-B/16, их PerceptualModel baseline=True); authors (их выпущенный dino_vitb16_single_lora);
каждый наш прогон — LoRA последней эпохи из runs/<имя>/.
Запуск (из ~/dreamsim_repro/code): python ../eval_b.py --models base authors human_s0 ... ; итог —
~/dreamsim_repro/results/eval_<модель>.json.
"""

import argparse
import glob
import io
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image
from scipy import stats
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

sys.path.insert(0, os.getcwd())  # код авторов (~/dreamsim_repro/code)
from dataset.dataset import TwoAFCDataset  # noqa: E402
from dreamsim import PerceptualModel, dreamsim  # noqa: E402
from evaluation.score import score_nights_dataset  # noqa: E402
from training.train import LightningPerceptualModel  # noqa: E402

HOME = os.path.expanduser("~/dreamsim_repro")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
LIMIT = int(os.environ.get("LIMIT", "0"))  # пробный запуск: столько пар из каждого набора
NIGHTS_ROOT = os.environ.get("NIGHTS_ROOT", "nights")
RES_DIR = os.path.join(HOME, os.environ.get("RES_DIR", "results"))
T = transforms.Compose([transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
                        transforms.ToTensor()])  # как get_preprocess_fn("DEFAULT", 224, BICUBIC)


class PairSet(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs  # (загрузчик эталона, загрузчик искажённого, оценка)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        a, b, s = self.pairs[i]
        return T(a().convert("RGB")), T(b().convert("RGB")), np.float32(s)


def kadid_pairs():
    d = os.path.join(HOME, "dataset", "iqa", "kadid10k")
    df = pd.read_csv(os.path.join(d, "dmos.csv"))
    im = os.path.join(d, "images")
    return [(lambda p=os.path.join(im, r.ref_img): Image.open(p), lambda p=os.path.join(im, r.dist_img): Image.open(p),
             r.dmos) for r in df.itertuples()]


def tid_pairs():
    rows = []
    for f in sorted(glob.glob(os.path.join(HOME, "dataset", "iqa", "tid2013_*.parquet"))):
        df = pd.read_parquet(f)
        for r in df.itertuples():
            rows.append((lambda b=r.reference["bytes"]: Image.open(io.BytesIO(b)),
                         lambda b=r.distorted["bytes"]: Image.open(io.BytesIO(b)), r.mos))
    return rows


@torch.no_grad()
def iqa(model, pairs, bs=32):
    dl = DataLoader(PairSet(pairs), batch_size=bs, num_workers=6, shuffle=False)
    dist, score = [], []
    for a, b, s in dl:
        dist.append(model(a.to(DEV), b.to(DEV)).float().cpu().reshape(-1)), score.append(s)
    d, s = torch.cat(dist).numpy(), torch.cat(score).numpy()
    return {"srocc": float(stats.spearmanr(-d, s)[0]), "plcc": float(stats.pearsonr(-d, s)[0]), "n": int(len(d))}


def load(name):
    if name == "base":
        return PerceptualModel(model_type="dino_vitb16", feat_type="cls", stride="16", baseline=True,
                               load_dir=os.path.join(HOME, "models"), device=DEV).to(DEV).eval()
    if name == "authors":
        m, _ = dreamsim(pretrained=True, device=DEV, cache_dir=os.path.join(HOME, "models"), dreamsim_type="dino_vitb16")
        return m.eval()
    run_dir = os.path.join(HOME, "runs", name)
    cfg_f = glob.glob(os.path.join(run_dir, "**", "config.yaml"), recursive=True)[0]
    cfg = vars(yaml.load(open(cfg_f), Loader=yaml.Loader))
    cfg["load_dir"] = os.path.join(HOME, "models")
    ep = sorted(glob.glob(os.path.join(run_dir, "**", "epoch_*", "adapter_config.json"), recursive=True),
                key=lambda p: int(p.split("epoch_")[-1].split(os.sep)[0]))[-1]
    lm = LightningPerceptualModel(**cfg, device=DEV)
    lm.load_lora_weights(os.path.dirname(ep))
    print(f"{name}: адаптер {os.path.dirname(ep)}", flush=True)
    return lm.perceptual_model.to(DEV).eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    args = ap.parse_args()
    os.makedirs(RES_DIR, exist_ok=True)
    sets = {"kadid10k": kadid_pairs(), "tid2013": tid_pairs()}
    if LIMIT:
        sets = {k: v[:: max(1, len(v) // LIMIT)][:LIMIT] for k, v in sets.items()}
    print({k: len(v) for k, v in sets.items()}, flush=True)
    nights = DataLoader(TwoAFCDataset(root_dir=os.path.join(HOME, "dataset", NIGHTS_ROOT), split="test"), batch_size=32,
                        num_workers=6, shuffle=False)
    for name in args.models:
        out_f = os.path.join(RES_DIR, f"eval_{name}.json")
        if os.path.exists(out_f):
            print("уже есть", out_f, flush=True)
            continue
        model = load(name)
        res = {"model": name, "nights_test_2afc": float(score_nights_dataset(model, nights, DEV))}
        for k, pairs in sets.items():
            res[k] = iqa(model, pairs)
        json.dump(res, open(out_f, "w"), indent=1)
        print(json.dumps(res), flush=True)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
