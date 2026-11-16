#!/usr/bin/env python
"""Проба непройденных понятий (CEM, приложение A.9): узкое место обученной модели -> 84 понятия, не вошедшие
в 28 обучаемых. Метки -- настоящие, из исходных train/val/test.pkl Koh et al.; изображения без аугментации
(центральный кадр 299, как у авторов на валидации).

Меры для каждого понятия (затем среднее по 84):
  acc_c1   -- точность LogisticRegression с настройками по умолчанию (C = 1) на сыром узком месте:
              «simple logistic linear model» авторов (точные настройки авторами не описаны);
  acc_tune -- признаки стандартизованы по train, C подобран на val из сетки -- полностью настроенная проба;
  ap_tune  -- средняя точность (AP) той же настроенной пробы (понятия с >= 5 положительными в тесте).
Ориентиры: тривиальный уровень (ответ большинства по train) и признаки исходной ResNet-34 (ImageNet, avgpool).
Средние -- в probe_results.jsonl, меры по каждому понятию (j -- номер среди 112 Koh) -- в probe_per_concept.jsonl.
Запуск: python probe_heldout.py [--folds 0 1 2 3 4] [--models cem hybrid noconcept placebo] [--pretrained]
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

REPO = os.path.expanduser("~/cem_repro/cem")
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.expanduser("~/cem_repro"))
import cem.train.training as training  # noqa: E402
from cem.data.CUB200 import cub_loader  # noqa: E402
import run_subsample_2022 as R  # noqa: E402

CS = [0.001, 0.01, 0.1, 1.0, 10.0]
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def loaders():
    """train/val/test без аугментации (имя файла без 'train.pkl' -> загрузчик не включает аугментацию)."""
    d = os.path.join(R.OUT, "probe_data")
    os.makedirs(d, exist_ok=True)
    out = {}
    for s in ("train", "val", "test"):
        link = os.path.join(d, f"{s}_noaug.pkl")
        if not os.path.exists(link):
            os.symlink(os.path.join(R.BASE_DIR, f"{s}.pkl"), link)
        out[s] = cub_loader.load_data(pkl_paths=[link], use_attr=True, no_img=False, batch_size=128,
                                      uncertain_label=False, n_class_attr=2, image_dir="images",
                                      resampling=False, root_dir=R.CUB_DIR, num_workers=3,
                                      concept_transform=np.asarray)  # список 112 меток -> массив
    return out


@torch.no_grad()
def embed(fn, dl):
    feats, ys, cs = [], [], []
    for x, y, c in dl:
        feats.append(fn(x.to(DEV)).float().cpu())
        ys.append(y), cs.append(c)
    return torch.cat(feats).numpy(), torch.cat(ys).numpy(), torch.cat(cs).numpy().astype(np.int64)


def probe(F, C, held):
    """F, C: словари split -> признаки / метки 112 понятий. Возвращает средние меры по held."""
    tr, va, te = F["train"], F["val"], F["test"]
    mu, sd = tr.mean(0), tr.std(0) + 1e-6
    trs, vas, tes = (tr - mu) / sd, (va - mu) / sd, (te - mu) / sd

    def one(j):
        ytr, yva, yte = C["train"][:, j], C["val"][:, j], C["test"][:, j]
        maj = int(ytr.mean() >= 0.5)
        triv = float((yte == maj).mean())
        if ytr.min() == ytr.max():
            return triv, triv, triv, None
        m1 = LogisticRegression(C=1.0, max_iter=5000).fit(tr, ytr)
        acc_c1 = float((m1.predict(te) == yte).mean())
        best = None
        for c in CS:
            m = LogisticRegression(C=c, max_iter=5000).fit(trs, ytr)
            v = float((m.predict(vas) == yva).mean())
            if best is None or v > best[0]:
                best = (v, m)
        m = best[1]
        acc_t = float((m.predict(tes) == yte).mean())
        ap = float(average_precision_score(yte, m.decision_function(tes))) if yte.sum() >= 5 else None
        return triv, acc_c1, acc_t, ap

    rows = Parallel(n_jobs=3)(delayed(one)(j) for j in held)
    aps = [r[3] for r in rows if r[3] is not None]
    per = [{"j": int(j), "trivial": r[0], "acc_c1": r[1], "acc_tune": r[2], "ap": r[3]} for j, r in zip(held, rows)]
    return {"trivial": float(np.mean([r[0] for r in rows])), "acc_c1": float(np.mean([r[1] for r in rows])),
            "acc_tune": float(np.mean([r[2] for r in rows])), "ap_tune": float(np.mean(aps)), "n_ap": len(aps)}, per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--models", nargs="+", default=["cem", "hybrid", "noconcept", "placebo"])
    ap.add_argument("--pretrained", action="store_true", help="ещё и признаки исходной ResNet-34")
    args = ap.parse_args()
    dls = loaders()
    out_f = os.path.join(R.OUT, "probe_results.jsonl")
    per_f = os.path.join(R.OUT, "probe_per_concept.jsonl")  # меры по каждому из 84 понятий

    def save(row, per):
        print(json.dumps(row), flush=True)
        open(out_f, "a").write(json.dumps(row) + "\n")
        open(per_f, "a").write(json.dumps({"model": row["model"], "fold": row["fold"], "concepts": per}) + "\n")
    if args.pretrained:
        import torchvision
        net = torchvision.models.resnet34(pretrained=True).to(DEV).eval()
        net.fc = torch.nn.Identity()
        E = {s: embed(net, dl) for s, dl in dls.items()}
        F, C = {s: E[s][0] for s in E}, {s: E[s][2] for s in E}
        for fold in args.folds:
            held = np.setdiff1d(np.arange(R.N_CONCEPTS), R.selected_concepts(fold))
            res, per = probe(F, C, held)
            save({"model": "pretrained_resnet34", "fold": fold, **res}, per)
        del net
    for model_name in args.models:
        for fold in args.folds:
            sel = R.selected_concepts(fold)
            cfg = R.model_config(model_name, len(sel))
            run = f"{cfg['architecture']}{cfg['extra_name']}_{cfg['c_extractor_arch']}_fold_{fold + 1}"
            path = os.path.join(R.OUT, f"{run}.pt")
            if not os.path.exists(path):
                print("нет модели", path, flush=True)
                continue
            # поправка на дисбаланс -- как при обучении (веса функции потерь входят в сохранённую модель)
            data_dir = R.data_dir_for(model_name, fold)
            imbalance = np.array(cub_loader.find_class_imbalance(R.imbalance_pkl(model_name, fold), True))[sel]
            model = training.construct_model(n_concepts=len(sel), n_tasks=R.N_TASKS, config=cfg,
                                             imbalance=imbalance)
            model.load_state_dict(torch.load(path, map_location="cpu"))
            model = model.to(DEV).eval()
            E = {s: embed(lambda x: model._forward(x, train=False)[1], dl) for s, dl in dls.items()}
            F, C = {s: E[s][0] for s in E}, {s: E[s][2] for s in E}
            with torch.no_grad():  # точность задачи (виды) на тесте -- проверка, что модель обучена
                logits = torch.cat([model._forward(x.to(DEV), train=False)[2].cpu() for x, _, _ in dls["test"]])
            task_acc = float((logits.argmax(1).numpy() == E["test"][1]).mean())
            held = np.setdiff1d(np.arange(R.N_CONCEPTS), sel)
            res, per = probe(F, C, held)
            save({"model": model_name, "fold": fold, "task_acc": task_acc, "dim": int(F["train"].shape[1]), **res},
                 per)
            del model
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
