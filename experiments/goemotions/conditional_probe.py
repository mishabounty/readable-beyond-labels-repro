#!/usr/bin/env python
"""Условная читаемость отложенных концептов B и связанность задачи с B (ridge-проба probe_ridge.probe, white=False).

Для каждого набора z (обычно обучаемые энкодеры, finetune_*):
  y   -- признаки: one-hot метка задачи; macro-AP B по ним -- связанность задачи с B;
  A   -- признаки: истинные метки концептов A; macro-AP B по ним -- насколько B выводится из A;
  z+A -- [z, метки A] (каждый блок центрируется и делится на свой скаляр); разность с A -- информация о B в z сверх
         меток A (условная проба в духе conditional V-information). Если выигрыш концептной супервизии -- только
         перенос через корреляцию A и B, то у bce эта разность не больше, чем у контролей без смысла;
  z   -- сам z (для сверки с probe_ridge_<набор>.jsonl), input -- вход до обучения (и input+A).
Результат: results/cond_probe_<набор>.jsonl; уже посчитанные строки пропускаются.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import train as T
from probe_eval import rescale, spec
from probe_ridge import SUFFIX, probe

HERE = Path(__file__).resolve().parent
EXP_DATA = HERE / "data"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sets", nargs="+")
    args = ap.parse_args()
    for name in args.sets:
        ds, ev, emb = spec(name)
        zdir = HERE / "results" / "z" / name
        x, y, split, M = T.load(ds, ev, 0, emb, False, 0)
        MA = T.load(ds, ev[:-1] + "A", 0, emb, False, 0)[3]
        if (zdir / "sub.npy").exists():
            sub = np.load(zdir / "sub.npy")
            if "_length" in name and ds == "goemotions":  # у reuters метка задачи и так квартиль длины
                from text_finetune import length_labels
                y = length_labels()
            for task in ("country",):  # естественные задачи: метки из data/<набор>/labels_<task>.npy
                if f"_{task}" in name:
                    y = torch.tensor(np.load(EXP_DATA / ds / f"labels_{task}.npy"), dtype=torch.long)
            x, y, M, MA = x[sub], y[sub], M[sub], MA[sub]
            nf, nv = (int(v) for v in np.load(zdir / "sub_split.npz").values()) if (zdir / "sub_split.npz").exists() \
                else (12000, 2000)
            split = {"fit": torch.arange(nf), "val": torch.arange(nf, nf + nv), "test": torch.arange(nf + nv, len(sub))}
        Y = M.toarray().astype(np.float32)
        fit = split["fit"]
        A = rescale(torch.tensor(MA.toarray(), dtype=torch.float64), fit)
        out = HERE / "results" / f"cond_probe{SUFFIX}_{name}.jsonl"
        done = {(r["method"], r["lambda"], r["seed"], r["features"]) for r in map(json.loads, open(out))} \
            if out.exists() else set()
        items = [("task_label", 0.0, 0, "y", rescale(F.one_hot(y).double(), fit)), ("concepts_A", 0.0, 0, "A", A)]
        xin = rescale(F.normalize(x, dim=-1).double(), fit)
        items += [("input", 0.0, 0, "z", xin), ("input", 0.0, 0, "z+A", torch.cat([xin, A], 1))]
        for f in sorted(zdir.glob("*_s*.npy")):
            method, lam, seed = f.stem.rsplit("_", 2)
            items += [(method, float(lam), int(seed[1:]), feats, f) for feats in ("z", "z+A")]
        for method, lam, seed, feats, src in items:
            if (method, lam, seed, feats) in done:
                continue
            if isinstance(src, Path):
                Z = rescale(torch.tensor(np.load(src).astype(np.float64)), fit)
                X = Z if feats == "z" else torch.cat([Z, A], 1)
            else:
                X = src
            m = probe(X.float().numpy(), Y, split, y, seed)
            row = {"set": name, "method": method, "lambda": lam, "seed": seed, "features": feats, **m}
            with open(out, "a") as fh:
                fh.write(json.dumps(row) + "\n")
            print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
