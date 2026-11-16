#!/usr/bin/env python
"""Перенос читаемости при малом числе меток: n-граммы против восстановления x и против сырого x.

Этап 1 (как в train.py, словарь ngram top-1000): task / bce / rank / recon (lambda 1 и 10).
Этап 2 на словаре --eval-dict (концепты людей, на этапе 1 не использовались):
  * full -- стандартный декодер train.stage2 (весь fit, эпоха по val AP);
  * n меток -- декодер на случайных n текстах fit, фиксированное число шагов без выбора по val
    (чтобы не подглядывать в большую валидацию), одинаково для всех представлений.
Ориентир -- тот же декодер на сыром x. Результат дописывается в results/fewshot_<датасет>.jsonl (поле train_dict; в старых строках его нет -- ngram).
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

import train as T

HERE = Path(__file__).resolve().parent


class Args:
    epochs, patience, batch, lr, gate_epochs = 40, 6, 256, 1e-3, 30


def fewshot_probe(z_all, split, M, n, seed, steps=300):
    g = torch.Generator().manual_seed(1000 + seed + n)
    fit = split["fit"][torch.randperm(len(split["fit"]), generator=g)[:n]]
    gate = T.concept_head(z_all.shape[1], M.shape[1]).to(T.DEV)
    opt = torch.optim.AdamW(gate.parameters(), lr=1e-2, weight_decay=0.01)
    zf, tf = z_all[fit].to(T.DEV), T.dense(M, fit).to(T.DEV)
    for _ in range(steps):
        b = torch.randint(len(fit), (min(len(fit), 1024),), generator=g).to(T.DEV)
        loss = T.bce_concepts(gate(zf[b]), tf[b])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    with torch.no_grad():
        return T.concept_metrics(gate(z_all[split["test"]].to(T.DEV)), T.dense(M, split["test"]).to(T.DEV))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--eval-dict", required=True)
    ap.add_argument("--k", type=int, default=1000)
    ap.add_argument("--train-dict", default="ngram", help="словарь этапа 1 (ngram_noaspect -- без утечки аспектов)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--shots", type=int, nargs="+", default=[50, 200, 1000])
    args = ap.parse_args()
    runs = [("raw_x", 0.0), ("task", 0.0), ("bce", 1.0), ("rank", 1.0), ("recon", 1.0), ("recon", 10.0)]
    for seed in args.seeds:
        _, _, _, M_eval = T.load(args.dataset, args.eval_dict, 0, "mqa", False, seed)
        x, y, split, M_train = T.load(args.dataset, args.train_dict, args.k, "mqa", False, seed)
        for method, lam in runs:
            torch.manual_seed(seed)
            gen = torch.Generator().manual_seed(seed)
            if method == "raw_x":
                z_all, task_m = F.normalize(x, dim=-1), {}
            else:
                enc, task_m = T.stage1(x, y, split, M_train, method, lam, Args, gen)
                with torch.no_grad():
                    z_all = torch.cat([enc(x[i:i + 4096].to(T.DEV)).cpu() for i in range(0, len(x), 4096)])
                del enc
                torch.cuda.empty_cache()
            res = {"full": T.stage2(z_all, split, M_eval, Args, gen)}
            for n in args.shots:
                res[str(n)] = fewshot_probe(z_all, split, M_eval, n, seed)
            row = {"dataset": args.dataset, "eval_dict": args.eval_dict, "method": method, "lambda": lam,
                   "train_dict": args.train_dict, "seed": seed, "task": task_m, "concepts": res}
            with open(HERE / "results" / f"fewshot_{args.dataset}.jsonl", "a") as fh:
                fh.write(json.dumps(row) + "\n")
            print(json.dumps({"ds": args.dataset, "seed": seed, "m": f"{method}{lam:g}",
                              "F1": round(task_m.get("test_macro_f1", float("nan")), 3),
                              **{k: round(v["ap"], 3) for k, v in res.items()}}), flush=True)


if __name__ == "__main__":
    main()
