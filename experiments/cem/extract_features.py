#!/usr/bin/env python
"""Признаки для дополнительных проверок опыта CEM (28.10.2026), без нового обучения.

Для каждой модели и разбиения за один проход по фото без аугментации (train / val / test) сохраняет:
  F — узкое место (то, что читает проба авторов): model._forward(x, train=False)[1];
  A — выход ResNet-34 перед узким местом (avgpool, 512 чисел) — одно и то же место у всех моделей, чтобы отделить
      «лучше представление» от «нелинейного узкого места» (у CEM узкое место нелинейное, у Hybrid — линейное);
  L — логиты задачи (вид), C — метки 112 атрибутов, Y — вид.
Файл: results/cub_subsample_0.25/features/<model>_fold<k>_full.npz.
С --pretrained ещё признаки исходной ResNet-34 (ImageNet) при её родной нормировке ImageNet: загрузчик Koh
нормирует как (x − 0.5) / 2, здесь вход переводится обратно и нормируется средним и разбросом ImageNet.
Файл: features/pretrained_resnet34_imagenet_full.npz.
Запуск: python extract_features.py --models cem placebo placebo2 --folds 0 1 2 3 4 [--pretrained]
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.expanduser("~/cem_repro"))
import probe_heldout as P  # noqa: E402

R = P.R
SPLITS = ("train", "val", "test")
IM_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IM_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def load_model(model_name, fold):
    sel = R.concepts_for(model_name, fold)
    cfg = R.model_config(model_name, len(sel))
    run = f"{cfg['architecture']}{cfg['extra_name']}_{cfg['c_extractor_arch']}_fold_{fold + 1}"
    data_dir = R.data_dir_for(model_name, fold)
    imb = np.array(P.cub_loader.find_class_imbalance(R.imbalance_pkl(model_name, fold), True))[sel]
    model = P.training.construct_model(n_concepts=len(sel), n_tasks=R.N_TASKS, config=cfg, imbalance=imb)
    model.load_state_dict(torch.load(os.path.join(R.OUT, f"{run}.pt"), map_location="cpu"))
    return model.to(P.DEV).eval()


@torch.no_grad()
def run_model(model, dl):
    pooled = {}
    av = [m for n, m in model.named_modules() if n.endswith("avgpool")]
    assert len(av) == 1, "ожидался ровно один avgpool"
    h = av[0].register_forward_hook(lambda m, i, o: pooled.__setitem__("a", o.flatten(1)))
    F, A, L, C, Y = [], [], [], [], []
    for x, y, c in dl:
        out = model._forward(x.to(P.DEV), train=False)
        F.append(out[1].float().cpu()), L.append(out[2].float().cpu()), A.append(pooled["a"].float().cpu())
        C.append(c), Y.append(y)
    h.remove()
    cat = lambda t: torch.cat(t).numpy()  # noqa: E731
    return cat(F), cat(A), cat(L), cat(C).astype(np.int8), cat(Y).astype(np.int16)


@torch.no_grad()
def run_pretrained(dl):
    import torchvision
    net = torchvision.models.resnet34(pretrained=True).to(P.DEV).eval()
    net.fc = torch.nn.Identity()
    F, C, Y = [], [], []
    for x, y, c in dl:
        x01 = x * 2.0 + 0.5  # обратно к [0, 1] из нормировки Koh (x − 0.5) / 2
        F.append(net(((x01 - IM_MEAN) / IM_STD).to(P.DEV)).float().cpu()), C.append(c), Y.append(y)
    cat = lambda t: torch.cat(t).numpy()  # noqa: E731
    return cat(F), cat(C).astype(np.int8), cat(Y).astype(np.int16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=[])
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--pretrained", action="store_true")
    args = ap.parse_args()
    out_dir = os.path.join(R.OUT, "features")
    os.makedirs(out_dir, exist_ok=True)
    dls = P.loaders()
    if args.pretrained:
        f = os.path.join(out_dir, "pretrained_resnet34_imagenet_full.npz")
        if not os.path.exists(f):
            res = {s: run_pretrained(dls[s]) for s in SPLITS}
            np.savez(f, **{f"{k}_{s}": v for s in SPLITS for k, v in zip(("F", "C", "Y"), res[s])})
            print("сохранено", f, flush=True)
    for m in args.models:
        for fold in args.folds:
            f = os.path.join(out_dir, f"{m}_fold{fold}_full.npz")
            if os.path.exists(f):
                print("уже есть", f, flush=True)
                continue
            model = load_model(m, fold)
            res = {s: run_model(model, dls[s]) for s in SPLITS}
            np.savez(f, **{f"{k}_{s}": v for s in SPLITS for k, v in zip(("F", "A", "L", "C", "Y"), res[s])})
            acc = float((res["test"][2].argmax(1) == res["test"][4]).mean())
            print(f"сохранено {f}; точность по видам на тесте {acc:.4f}", flush=True)
            del model
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
