#!/usr/bin/env python
"""Обучаемый энкодер на CUB: снимает возражение «при замороженных признаках лучшее -- не разрушить вход».

Энкодер: последний блок ResNet-18 (layer4, инициализация ImageNet) дообучается поверх замороженных выходов layer3
(embed_cub_resnet.py), затем пулинг и MLP как в train.py: z = normalize(MLP(pool(layer4(h)))).
Этап 1 -- задача (--dataset cub: 200 видов, cub4: 4 надкласса) плюс вспомогательная потеря на атрибутах A
(cubA, уровень изображений): task / bce / rank / rand (случайные атрибуты той же плотности) / recon (восстановление
исходных признаков предобученного layer4) / unif (равномерность z). Ранняя остановка по val macro-F1.
Читаемость отложенных атрибутов B и меры схлопывания -- как в collapse_check.py. Ориентир raw_x -- исходные
признаки предобученного layer4 (вход до дообучения).
Результат дописывается в results/finetune_<датасет>.jsonl.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

import train as T
from collapse_check import collapse_metrics, readability
from transfer_fewshot import Args, fewshot_probe

HERE = Path(__file__).resolve().parent
WEIGHTS = Path.home() / ".cache/torch/hub/checkpoints/resnet18-f37072fd.pth"


class BasicBlock(nn.Module):
    """Блок ResNet с теми же именами параметров, что в torchvision (чтобы загрузить веса ImageNet)."""

    def __init__(self, cin, cout, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.downsample = (nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))
                           if stride != 1 or cin != cout else None)

    def forward(self, h):
        out = F.relu(self.bn1(self.conv1(h)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + (h if self.downsample is None else self.downsample(h)))


class Layer4Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(BasicBlock(256, 512, 2), BasicBlock(512, 512, 1))  # layer4 ResNet-18
        state = torch.load(WEIGHTS, map_location="cpu")
        self.backbone.load_state_dict({k[len("layer4."):]: v for k, v in state.items() if k.startswith("layer4.")})
        self.mlp = T.Encoder(512)

    def forward(self, h):
        return self.mlp(self.backbone(h).mean((2, 3)))


def memmap_fetch(maps):
    """Входы энкодера для набора индексов (по возрастанию -- быстрее читать memmap)."""
    return lambda idx: (torch.from_numpy(np.asarray(maps[idx], dtype=np.float32)).to(T.DEV),)


def run(fetch, make_encoder, x, y, split, M, method, lam, gen, epochs=15, patience=4, bs=64):
    """Этап 1 с обучаемым верхом энкодера: fetch(индексы) -> входы энкодера, make_encoder() -> модуль с частями
    backbone (дообучаемый верх предобученной сети) и mlp.
    Методы против забывания (только задача): l2sp -- штраф lam * |theta - theta0|^2 на веса backbone (L2-SP);
    lpft -- первые epochs // 3 эпох backbone заморожен (линейная проба, затем дообучение); wise -- после обучения
    веса backbone усредняются с предобученными 0.5 / 0.5 (WiSE-FT). aonly -- только BCE на концептах M, без задачи:
    голова задачи учится на z.detach() (для отчёта), эпоха выбирается по BCE концептов на валидации."""
    n_cls = int(y.max()) + 1
    enc, task = make_encoder().to(T.DEV), nn.Linear(768, n_cls).to(T.DEV)
    theta0 = {k: v.detach().clone() for k, v in enc.backbone.state_dict().items()}  # предобученные веса
    lp_epochs = max(1, epochs // 3) if method == "lpft" else 0
    if method in ("bce", "bce_bal", "rank", "rand", "aonly"):
        aux = T.concept_head(768, M.shape[1]).to(T.DEV)
    elif method == "recon":
        aux = nn.Linear(768, x.shape[1]).to(T.DEV)
    else:
        aux = None
    groups = [{"params": enc.backbone.parameters(), "lr": 1e-4},
              {"params": list(enc.mlp.parameters()) + list(task.parameters()), "lr": 1e-3}]
    if aux is not None:
        groups.append({"params": aux.parameters(), "lr": 1e-2})
    opt = torch.optim.AdamW(groups, weight_decay=0.01)

    @torch.no_grad()
    def embed(idx):  # idx по возрастанию
        enc.eval()
        return torch.cat([enc(*fetch(idx[i:i + 256].numpy())).cpu() for i in range(0, len(idx), 256)])

    best, stale = None, 0
    for ep in range(1, epochs + 1):
        enc.train()
        for b in T.batches(split["fit"], bs, True, gen):
            b = torch.sort(b).values
            z = enc(*fetch(b.numpy()))
            loss = F.cross_entropy(10.0 * task(z.detach() if method == "aonly" else z), y[b].to(T.DEV))
            if method == "aonly":
                loss = loss + lam * T.bce_concepts(aux(z), T.dense(M, b).to(T.DEV))
            elif method == "l2sp":
                loss = loss + lam * sum(((p - theta0[n]) ** 2).sum() for n, p in enc.backbone.named_parameters())
            if method in ("bce", "rand"):
                loss = loss + lam * T.bce_concepts(aux(z), T.dense(M, b).to(T.DEV))
            elif method == "bce_bal":
                loss = loss + lam * T.concept_loss("bce_bal", aux(z), T.dense(M, b).to(T.DEV))
            elif method == "vicreg":
                loss = loss + lam * T.vicreg_loss(z)
            elif method == "rank":
                loss = loss + lam * T.rank_concepts(aux(z), T.dense(M, b).to(T.DEV))
            elif method == "recon":
                loss = loss + lam * (1 - F.cosine_similarity(aux(z), x[b].to(T.DEV), dim=-1)).mean()
            elif method == "unif":
                loss = loss + lam * torch.pdist(z).pow(2).mul(-2).exp().mean().log()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if ep <= lp_epochs:  # LP-FT: сначала обучается только голова (MLP и задача)
                for p_ in enc.backbone.parameters():
                    p_.grad = None
            opt.step()
        if ep <= lp_epochs:
            continue
        val = torch.sort(split["val"]).values
        with torch.no_grad():
            zv = embed(val).to(T.DEV)
            if method == "aonly":  # выбор эпохи по BCE концептов (задача на z не влияет)
                f1 = -float(T.bce_concepts(aux(zv), T.dense(M, val).to(T.DEV)))
            else:
                pv = (10.0 * task(zv)).argmax(1).cpu()
                f1 = T.macro_f1(y[val], pv, n_cls)
        if best is None or f1 > best[0] + 1e-4:
            best, stale = (f1, ep, {k: v.detach().clone() for k, v in enc.state_dict().items()},
                           {k: v.detach().clone() for k, v in task.state_dict().items()}), 0
        else:
            stale += 1
            if stale >= patience:
                break
    enc.load_state_dict(best[2])
    task.load_state_dict(best[3])
    if method == "wise":  # WiSE-FT: середина между предобученными и дообученными весами backbone
        sd = enc.backbone.state_dict()
        enc.backbone.load_state_dict({k: (0.5 * theta0[k] + 0.5 * v) if v.is_floating_point() else v
                                      for k, v in sd.items()})
    z_all = embed(torch.arange(len(y)))
    te = split["test"]
    with torch.no_grad():
        pt = (10.0 * task(z_all[te].to(T.DEV))).argmax(1).cpu()
    del enc, task, aux, opt
    torch.cuda.empty_cache()
    return z_all, {"best_epoch": best[1], "test_macro_f1": T.macro_f1(y[te], pt, n_cls)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cub4")
    ap.add_argument("--concepts", default="cub", help="словари <concepts>A / <concepts>B (sun -- SUN Attributes)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--runs", nargs="+", default=["task:0", "bce:1", "rank:1", "rand:1", "recon:3", "unif:1"])
    ap.add_argument("--out-tag", default="", help="суффикс файла результатов и папки z")
    ap.add_argument("--save-z", action="store_true", help="сохранить z в results/z/finetune_<датасет>/")
    ap.add_argument("--no-probe", action="store_true", help="без прежней пробы (читаемость -- probe_eval / probe_ridge)")
    args = ap.parse_args()
    d = HERE / "data" / args.dataset
    maps = np.load(d / "resnet18_layer3.npy", mmap_mode="r")
    out = HERE / "results" / f"finetune_{args.dataset}{args.out_tag}.jsonl"
    for seed in args.seeds:
        cA, cB = f"{args.concepts}A", f"{args.concepts}B"
        _, _, _, M_eval = T.load(args.dataset, cB, 0, "resnet18", False, seed)
        x, y, split, M_A = T.load(args.dataset, cA, 0, "resnet18", False, seed)
        M_rand = T.load(args.dataset, cA, 0, "resnet18", True, seed)[3]
        if seed == args.seeds[0] and not args.no_probe:
            gen = torch.Generator().manual_seed(0)
            xn = F.normalize(x, dim=-1)
            row = {"dataset": args.dataset, "method": "raw_x", "lambda": 0.0, "seed": 0, "task_f1": None,
                   **collapse_metrics(xn[split["test"]], y[split["test"]]), **readability(xn, split, M_eval, y, gen),
                   "ap_200": fewshot_probe(xn, split, M_eval, 200, 0)["ap"]}
            with open(out, "a") as fh:
                fh.write(json.dumps(row) + "\n")
            print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)
        for r in args.runs:
            method, lam = r.split(":")[0], float(r.split(":")[1])
            torch.manual_seed(seed)
            gen = torch.Generator().manual_seed(seed)
            t0 = time.time()
            z_all, task_m = run(memmap_fetch(maps), Layer4Encoder, x, y, split, M_rand if method == "rand" else M_A,
                                method, lam, gen)
            te = split["test"]
            if args.save_z:
                zdir = HERE / "results" / "z" / f"finetune_{args.dataset}{args.out_tag}"
                zdir.mkdir(parents=True, exist_ok=True)
                np.save(zdir / f"{method}_{lam:g}_s{seed}.npy", z_all.numpy().astype(np.float16))
            row = {"dataset": args.dataset, "method": method, "lambda": lam, "seed": seed,
                   "task_f1": task_m["test_macro_f1"], "best_epoch": task_m["best_epoch"],
                   **collapse_metrics(z_all[te], y[te])}
            if not args.no_probe:
                row.update(readability(z_all, split, M_eval, y, gen),
                           ap_200=fewshot_probe(z_all, split, M_eval, 200, seed)["ap"])
            row["seconds"] = time.time() - t0
            with open(out, "a") as fh:
                fh.write(json.dumps(row) + "\n")
            print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)


if __name__ == "__main__":
    main()
