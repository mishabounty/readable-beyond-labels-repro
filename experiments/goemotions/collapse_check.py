#!/usr/bin/env python
"""Решающая проверка тезиса «концептная потеря работает как защита от схлопывания».

Этап 1 (как в train.py) с разными вспомогательными потерями и весами; словарь концептов этапа 1 -- n-граммы
без утечки концептов людей (CEBaB: ngram_noaspect, GoEmotions: ngram_noemotion, top-1000):
  task; bce / rank -- концепты; recon -- восстановление x; rand -- BCE на случайных концептах той же плотности;
  unif -- равномерность на сфере (ни концептов, ни входа);
  cbm / cem -- опубликованные концептные модели (concept_models.py), *_rand -- они же на случайных концептах.
Для каждого прогона на тесте:
  схлопывание -- эффективный ранг z (exp энтропии нормированных сингулярных чисел центрированной матрицы) и доля
                 разброса внутри классов задачи tr(S_w) / tr(S_t);
  читаемость  -- AP концептов людей (aspect / emotion, на этапе 1 не участвуют) свежим декодером на всех метках и
                 на 200 метках; auc_within -- AUC того же декодера внутри класса задачи (различает ли он примеры
                 одного класса с концептом и без; пары класс--концепт с >= 5 примерами с концептом и без).
Результат дописывается в results/collapse_<датасет>.jsonl.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import concept_models as CM
import train as T
from transfer_fewshot import Args, fewshot_probe

HERE = Path(__file__).resolve().parent
TRAIN_DICT = {"cebab": "ngram_noaspect", "goemotions": "ngram_noemotion"}
EVAL_DICT = {"cebab": "aspect", "goemotions": "emotion"}
RUNS = [("task", 0.0), ("bce", 0.1), ("bce", 0.3), ("bce", 1.0), ("bce", 3.0), ("rank", 0.3), ("rank", 1.0),
        ("recon", 1.0), ("recon", 3.0), ("recon", 10.0), ("rand", 1.0), ("rand", 3.0),
        ("unif", 0.1), ("unif", 1.0), ("unif", 10.0),
        ("cbm", 1.0), ("cbm_rand", 1.0), ("cem", 1.0), ("cem_rand", 1.0)]


@torch.no_grad()
def within_class_auc(scores, target, y, min_count=5):
    aucs = []
    for c in y.unique():
        S, T_ = scores[y == c], target[y == c].float()
        pos, neg = T_.sum(0), (1 - T_).sum(0)
        ok = (pos >= min_count) & (neg >= min_count)
        if ok.any():
            ranks = S[:, ok].argsort(0).argsort(0).float() + 1
            aucs.append(((ranks * T_[:, ok]).sum(0) - pos[ok] * (pos[ok] + 1) / 2) / (pos[ok] * neg[ok]))
    return float(torch.cat(aucs).mean()) if aucs else float("nan")


@torch.no_grad()
def collapse_metrics(z, y):
    zc = z - z.mean(0)
    s = torch.linalg.svdvals(zc)
    p = s / s.sum()
    erank = float(torch.exp(-(p * torch.log(p.clamp_min(1e-12))).sum()))
    within = sum(((z[y == c] - z[y == c].mean(0)) ** 2).sum() for c in y.unique())
    return {"effective_rank": erank, "within_class_share": float(within / (zc ** 2).sum())}


def readability(z_all, split, M_eval, y, gen):
    m, scores = T.stage2(z_all, split, M_eval, Args, gen, return_scores=True)
    te = split["test"]
    return {"ap_full": m["ap"], "auc_within": within_class_auc(scores, T.dense(M_eval, te), y[te])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["cebab", "goemotions"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--methods", nargs="+", help="только эти методы (по умолчанию все)")
    ap.add_argument("--no-raw", action="store_true", help="не считать ориентир на сыром x")
    ap.add_argument("--train-dict", help="словарь этапа 1 вместо TRAIN_DICT (например, emotionA)")
    ap.add_argument("--eval-dict", help="словарь проверки вместо EVAL_DICT (например, emotionB)")
    ap.add_argument("--k", type=int, default=1000, help="размер словаря этапа 1, 0 -- весь")
    ap.add_argument("--runs", nargs="+", help="метод:lambda вместо RUNS")
    ap.add_argument("--emb", default="mqa", help="признаки data/<ds>/emb_<emb>.npy (cub: inception)")
    ap.add_argument("--save-z", action="store_true",
                    help="сохранить z каждого прогона в results/z/<датасет><out-tag>/<метод>_<lambda>_s<seed>.npy")
    ap.add_argument("--out-tag", default="", help="результаты в results/collapse_<датасет><out-tag>.jsonl")
    ap.add_argument("--logit-scale", type=float, default=10.0, help="масштаб логитов задачи (ручка схлопывания)")
    ap.add_argument("--label-smoothing", type=float, default=0.0)
    ap.add_argument("--wd", type=float, default=0.01, help="weight decay этапа 1")
    ap.add_argument("--fixed-epochs", type=int, default=0, help="> 0 -- ровно столько эпох без ранней остановки")
    ap.add_argument("--no-probe", action="store_true",
                    help="без пробы этапа 2 (читаемость затем считает probe_eval.py по сохранённым z)")
    args = ap.parse_args()
    runs = [(r.split(":")[0], float(r.split(":")[1])) for r in args.runs] if args.runs else RUNS
    knobs = {"logit_scale": args.logit_scale, "label_smoothing": args.label_smoothing, "wd": args.wd,
             "fixed_epochs": args.fixed_epochs}
    for k, v in knobs.items():
        setattr(Args, k, v)
    for ds in args.datasets:
        for seed in args.seeds:
            train_dict, eval_dict = args.train_dict or TRAIN_DICT[ds], args.eval_dict or EVAL_DICT[ds]
            _, _, _, M_eval = T.load(ds, eval_dict, 0, args.emb, False, seed)
            x, y, split, M_ng = T.load(ds, train_dict, args.k, args.emb, False, seed)
            M_rand = T.load(ds, train_dict, args.k, args.emb, True, seed)[3]
            for method, lam in runs:
                if args.methods and method not in args.methods:
                    continue
                torch.manual_seed(seed)
                gen = torch.Generator().manual_seed(seed)
                t0 = time.time()
                M = M_rand if "rand" in method else M_ng
                if method.startswith(("cbm", "cem")):
                    enc, task_m = CM.stage1(x, y, split, M, method[:3], lam, Args, gen)
                else:  # rand -- bce на случайных концептах, rand_bal -- bce_bal на них же
                    m1 = {"rand": "bce", "rand_bal": "bce_bal"}.get(method, method)
                    enc, task_m = T.stage1(x, y, split, M, m1, lam, Args, gen)
                with torch.no_grad():
                    z_all = torch.cat([enc(x[i:i + 4096].to(T.DEV)).cpu() for i in range(0, len(x), 4096)])
                del enc
                torch.cuda.empty_cache()
                if args.save_z:
                    zdir = HERE / "results" / "z" / f"{ds}{args.out_tag}"
                    zdir.mkdir(parents=True, exist_ok=True)
                    np.save(zdir / f"{method}_{lam:g}_s{seed}.npy", z_all.numpy().astype(np.float16))
                te = split["test"]
                row = {"dataset": ds, "method": method, "lambda": lam, "seed": seed,
                       "task_f1": task_m["test_macro_f1"], "best_epoch": task_m["best_epoch"],
                       "a_ap": task_m.get("a_ap"), "a_macro_ap": task_m.get("a_macro_ap"),
                       **{k: v for k, v in knobs.items() if v != getattr(T, "DEFAULT_KNOBS")[k]},
                       **collapse_metrics(z_all[te], y[te])}
                if not args.no_probe:
                    row.update(readability(z_all, split, M_eval, y, gen),
                               ap_200=fewshot_probe(z_all, split, M_eval, 200, seed)["ap"])
                row["seconds"] = time.time() - t0
                with open(HERE / "results" / f"collapse_{ds}{args.out_tag}.jsonl", "a") as fh:
                    fh.write(json.dumps(row) + "\n")
                print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}),
                      flush=True)
        if args.no_raw:
            continue
        # ориентир: сырой x
        x, y, split, _ = T.load(ds, args.train_dict or TRAIN_DICT[ds], args.k, args.emb, False, 0)
        gen = torch.Generator().manual_seed(0)
        xn = F.normalize(x, dim=-1)
        row = {"dataset": ds, "method": "raw_x", "lambda": 0.0, "seed": 0, "task_f1": None,
               **collapse_metrics(xn[split["test"]], y[split["test"]]),
               **readability(xn, split, M_eval, y, gen),
               "ap_200": fewshot_probe(xn, split, M_eval, 200, 0)["ap"]}
        with open(HERE / "results" / f"collapse_{ds}{args.out_tag}.jsonl", "a") as fh:
            fh.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
