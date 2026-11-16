#!/usr/bin/env python
"""Варианты «простой логистической линейной модели» авторов CEM (A.9) на узком месте.

Код пробы авторы не выложили (в коммите 26f027dd6f0c его нет). У нас на разбиении 0 разрыв CEM − Hybrid 0.6 п.
против их 2.5 п. Проверяем одну простую причину: LogisticRegression со значениями sklearn по умолчанию
(max_iter = 100) на сырых числах может не сойтись, особенно на неограниченных логитах Hybrid. Считаем точность
C = 1 при max_iter = 100 и при 5000 (наша проба 1) и число несошедшихся атрибутов; рядом — масштаб признаков.

Признаки сохраняются: results/cub_subsample_0.25/features/<model>_fold<k>.npz (train/val/test, float32) —
для дальнейших пересчётов без GPU. Итог: probe_variants.jsonl.
Запуск: python probe_variants.py --folds 0 --models pretrained_resnet34 cem placebo hybrid noconcept
"""

import argparse
import json
import os
import sys
import warnings

import numpy as np
import torch
from joblib import Parallel, delayed
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.expanduser("~/cem_repro"))
import probe_heldout as P  # noqa: E402

R = P.R
SPLITS = ("train", "val", "test")


def features(model_name, fold, dls):
    """Признаки узкого места (или avgpool исходной ResNet-34) и метки 112 понятий; кэш на диске."""
    d = os.path.join(R.OUT, "features")
    os.makedirs(d, exist_ok=True)
    tag = "pretrained_resnet34" if model_name == "pretrained_resnet34" else f"{model_name}_fold{fold}"
    f = os.path.join(d, f"{tag}.npz")
    if os.path.exists(f):
        z = np.load(f)
        return {s: z[f"F_{s}"] for s in SPLITS}, {s: z[f"C_{s}"] for s in SPLITS}
    if model_name == "pretrained_resnet34":
        import torchvision
        net = torchvision.models.resnet34(pretrained=True).to(P.DEV).eval()
        net.fc = torch.nn.Identity()
        fn = net
    else:
        sel = R.selected_concepts(fold)
        cfg = R.model_config(model_name, len(sel))
        run = f"{cfg['architecture']}{cfg['extra_name']}_{cfg['c_extractor_arch']}_fold_{fold + 1}"
        data_dir = R.BASE_DIR if model_name != "placebo" else \
            os.path.join(R.OUT, "placebo_data", f"fold_{fold}")
        imb = np.array(P.cub_loader.find_class_imbalance(os.path.join(data_dir, "train.pkl"), True))[sel]
        model = P.training.construct_model(n_concepts=len(sel), n_tasks=R.N_TASKS, config=cfg, imbalance=imb)
        model.load_state_dict(torch.load(os.path.join(R.OUT, f"{run}.pt"), map_location="cpu"))
        model = model.to(P.DEV).eval()
        fn = lambda x: model._forward(x, train=False)[1]  # noqa: E731
    E = {s: P.embed(fn, dl) for s, dl in dls.items()}
    F = {s: E[s][0].astype(np.float32) for s in SPLITS}
    C = {s: E[s][2] for s in SPLITS}
    np.savez(f, **{f"F_{s}": F[s] for s in SPLITS}, **{f"C_{s}": C[s] for s in SPLITS})
    return F, C


def logreg_c1(F, C, held, max_iter):
    """Средняя по held точность на test LogisticRegression(C = 1) на сырых признаках и число несошедшихся."""
    def one(j):
        ytr, yte = C["train"][:, j], C["test"][:, j]
        if ytr.min() == ytr.max():
            return float((yte == ytr[0]).mean()), True
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", ConvergenceWarning)
            m = LogisticRegression(C=1.0, max_iter=max_iter).fit(F["train"], ytr)
        conv = not any(issubclass(x.category, ConvergenceWarning) for x in w)
        return float((m.predict(F["test"]) == yte).mean()), conv

    rows = Parallel(n_jobs=3)(delayed(one)(j) for j in held)
    return float(np.mean([r[0] for r in rows])), int(sum(not r[1] for r in rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, nargs="+", default=[0])
    ap.add_argument("--models", nargs="+", default=["pretrained_resnet34", "cem", "placebo", "hybrid", "noconcept"])
    args = ap.parse_args()
    dls = P.loaders()
    out_f = os.path.join(R.OUT, "probe_variants.jsonl")
    for fold in args.folds:
        held = np.setdiff1d(np.arange(R.N_CONCEPTS), R.selected_concepts(fold))
        for m in args.models:
            F, C = features(m, fold, dls)
            row = {"model": m, "fold": fold, "dim": int(F["train"].shape[1]),
                   "mean_abs": float(np.abs(F["train"]).mean()), "max_abs": float(np.abs(F["train"]).max())}
            for it in (100, 5000):
                acc, n_bad = logreg_c1(F, C, held, it)
                row[f"acc_c1_it{it}"], row[f"not_conv_it{it}"] = acc, n_bad
            print(json.dumps(row), flush=True)
            open(out_f, "a").write(json.dumps(row) + "\n")
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
