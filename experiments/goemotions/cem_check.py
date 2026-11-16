#!/usr/bin/env python
"""Быстрая проверка опыта CEM §5.3 по нашему протоколу (CUB, учат 28 атрибутов вида, читают остальные).

Энкодер как в cub_finetune.py: дообучается layer4 ResNet-18 поверх замороженного layer3, затем MLP, z (768).
Методы:
  task -- только задача (200 видов), линейная голова на z;
  cem  -- голова CEM (concept_models.CEMHead, m=28 концептов, вложения по 16): задача по узкому месту
          (смесь вложений концептов, 28 x 16 = 448), потеря CE + lam * BCE концептов;
  hcbm -- Hybrid-CBM: узкое место [sigmoid(логиты 28 концептов), 420 свободных нейронов] той же ширины 448.
Концепты задаются --concepts (cemA -- настоящие атрибуты, cemArnd -- пустышка той же формы).
Сохраняются z и узкое место (float32) в results/z/cem_check/ для чтения cem_readout.py.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

import concept_models as CM
import train as T
from cub_finetune import Layer4Encoder, memmap_fetch

HERE = Path(__file__).resolve().parent
ZDIR = HERE / "results" / "z" / "cem_check"
SCALE = 768 ** 0.5  # z нормирован на единицу; голова концептной модели получает z * sqrt(768) (признаки порядка 1)


class CEMB(CM.CEMHead):
    """CEM, который отдаёт ещё и узкое место (смесь вложений концептов)."""

    def forward(self, z):
        c = F.leaky_relu(self.embed(z)).view(len(z), self.m, 2, self.emb)
        logits = self.score(c.flatten(2)).squeeze(-1)
        p = torch.sigmoid(logits)[..., None]
        mixed = (p * c[:, :, 0] + (1 - p) * c[:, :, 1]).flatten(1)
        return logits, self.task(mixed), mixed


class HybridCBM(nn.Module):
    def __init__(self, m, n_cls, width=448):
        super().__init__()
        self.concepts = T.concept_head(768, m)
        self.extra = nn.Linear(768, width - m)
        self.task = nn.Linear(width, n_cls)

    def forward(self, z):
        logits = self.concepts(z)
        b = torch.cat([torch.sigmoid(logits), F.leaky_relu(self.extra(z))], 1)
        return logits, self.task(b), b


def run(fetch, y, split, M, method, lam, gen, epochs=40, patience=6, bs=64, head_lr=1e-3):
    n_cls = int(y.max()) + 1
    enc = Layer4Encoder().to(T.DEV)
    if method == "task":
        head = nn.Linear(768, n_cls).to(T.DEV)
    else:
        head = (CEMB(M.shape[1], n_cls) if method == "cem" else HybridCBM(M.shape[1], n_cls)).to(T.DEV)
    opt = torch.optim.AdamW([{"params": enc.backbone.parameters(), "lr": 1e-4},
                             {"params": enc.mlp.parameters(), "lr": 1e-3},
                             {"params": head.parameters(), "lr": head_lr}],
                            weight_decay=0.01)

    def forward(z):  # (логиты концептов или None, выход задачи, узкое место или None)
        return (None, 10.0 * head(z), None) if method == "task" else head(SCALE * z)

    @torch.no_grad()
    def embed(idx):
        enc.eval()
        zs, outs, bs_, lgs = [], [], [], []
        for i in range(0, len(idx), 256):
            z = enc(*fetch(idx[i:i + 256].numpy()))
            lg, out, b = forward(z)
            zs.append(z.cpu()), outs.append(out.cpu())
            bs_.append((b if b is not None else z[:, :0]).cpu())
            lgs.append((lg if lg is not None else z[:, :0]).cpu())
        return torch.cat(zs), torch.cat(outs), torch.cat(bs_), torch.cat(lgs)

    best, stale = None, 0
    for ep in range(1, epochs + 1):
        enc.train()
        for b in T.batches(split["fit"], bs, True, gen):
            b = torch.sort(b).values
            lg, out, _ = forward(enc(*fetch(b.numpy())))
            loss = F.cross_entropy(out, y[b].to(T.DEV))
            if lg is not None:
                loss = loss + lam * T.bce_concepts(lg, T.dense(M, b).to(T.DEV))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        val = torch.sort(split["val"]).values
        f1 = T.macro_f1(y[val], embed(val)[1].argmax(1), n_cls)
        if best is None or f1 > best[0] + 1e-4:
            best, stale = (f1, ep, {k: v.detach().clone() for k, v in enc.state_dict().items()},
                           {k: v.detach().clone() for k, v in head.state_dict().items()}), 0
        else:
            stale += 1
            if stale >= patience:
                break
    enc.load_state_dict(best[2])
    head.load_state_dict(best[3])
    z_all, out_all, b_all, lg_all = embed(torch.arange(len(y)))
    te = split["test"]
    m = {"best_epoch": best[1], "test_macro_f1": T.macro_f1(y[te], out_all[te].argmax(1), n_cls),
         "test_acc": float((out_all[te].argmax(1) == y[te]).float().mean())}
    if lg_all.shape[1]:  # насколько выучены сами концепты A (AP логитов на тесте)
        a_m = T.concept_metrics(lg_all[te], T.dense(M, te))
        m.update({f"A_{k}": v for k, v in a_m.items() if isinstance(v, float)})
    del enc, head, opt
    torch.cuda.empty_cache()
    return z_all, b_all, m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["task:0:cemA", "cem:1:cemA", "cem:1:cemArnd", "hcbm:1:cemA"],
                    help="метод:lambda:словарь концептов A")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--head-lr", type=float, default=1e-3, help="скорость обучения головы (задачи или CEM/CBM)")
    ap.add_argument("--tag", default="", help="суффикс файла результатов и папки z")
    args = ap.parse_args()
    d = HERE / "data" / "cub"
    maps = np.load(d / "resnet18_layer3.npy", mmap_mode="r")
    zdir = ZDIR.parent / f"cem_check{args.tag}"
    zdir.mkdir(parents=True, exist_ok=True)
    out = HERE / "results" / f"cem_check{args.tag}.jsonl"
    for seed in args.seeds:
        for r in args.runs:
            parts = r.split(":")  # метод:lambda:концепты[:epN -- ровно N эпох]
            method, lam, conc = parts[:3]
            n_ep = int(parts[3][2:]) if len(parts) > 3 else 0
            _, y, split, M = T.load("cub", conc, 0, "resnet18", False, seed)
            torch.manual_seed(seed)
            gen = torch.Generator().manual_seed(seed)
            t0 = time.time()
            z_all, b_all, m = run(memmap_fetch(maps), y, split, M, method, float(lam), gen,
                                  epochs=n_ep or args.epochs, patience=n_ep or args.patience, head_lr=args.head_lr)
            tag = f"{method}{float(lam):g}{f'_ep{n_ep}' if n_ep else ''}_{conc}_s{seed}"
            np.save(zdir / f"{tag}_z.npy", z_all.numpy().astype(np.float32))
            if b_all.shape[1]:
                np.save(zdir / f"{tag}_b.npy", b_all.numpy().astype(np.float32))
            row = {"tag": tag.rsplit("_s", 1)[0], "method": method, "lambda": float(lam), "concepts": conc, "seed": seed,
                   "head_lr": args.head_lr,
                   "epochs": args.epochs, "patience": args.patience, **m,
                   "seconds": time.time() - t0}
            with open(out, "a") as fh:
                fh.write(json.dumps(row) + "\n")
            print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)


if __name__ == "__main__":
    main()
