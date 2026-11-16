#!/usr/bin/env python
"""Перенос читаемости: обучение z с бесплатными концептами (n-граммы), чтение ДРУГИХ концептов (разметка людей).

Этап 1: как в train.py, вспомогательная потеря на словаре --train-dict (обычно ngram top-k).
Этап 2: свежий декодер на словаре --eval-dict (emotion для GoEmotions, aspect для CEBaB), который на этапе 1
не использовался. Результат дописывается в results/transfer_<датасет>.jsonl.
"""

import argparse
import json
import time
from pathlib import Path

import torch

import train as T

HERE = Path(__file__).resolve().parent


class Args:
    epochs, patience, batch, lr, gate_epochs = 40, 6, 256, 1e-3, 30


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--eval-dict", required=True)
    ap.add_argument("--k", type=int, nargs="+", default=[1000, 5000])
    ap.add_argument("--methods", nargs="+", default=["task", "bce", "rank"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    args = ap.parse_args()
    for seed in args.seeds:
        _, _, _, M_eval = T.load(args.dataset, args.eval_dict, 0, "mqa", False, seed)
        for k in args.k:
            x, y, split, M_train = T.load(args.dataset, "ngram", k, "mqa", False, seed)
            for method in args.methods:
                if method == "task" and k != args.k[0]:
                    continue  # обучение только на задачу от словаря не зависит
                torch.manual_seed(seed)
                gen = torch.Generator().manual_seed(seed)
                t0 = time.time()
                enc, task_m = T.stage1(x, y, split, M_train, method, 1.0, Args, gen)
                with torch.no_grad():
                    z_all = torch.cat([enc(x[i:i + 4096].to(T.DEV)).cpu() for i in range(0, len(x), 4096)])
                del enc
                torch.cuda.empty_cache()
                m = T.stage2(z_all, split, M_eval, Args, gen)
                row = {"dataset": args.dataset, "train_dict": f"ngram{k}" if method != "task" else "—",
                       "eval_dict": args.eval_dict, "method": method, "seed": seed, "task": task_m, "concepts": m,
                       "seconds": time.time() - t0}
                with open(HERE / "results" / f"transfer_{args.dataset}.jsonl", "a") as fh:
                    fh.write(json.dumps(row) + "\n")
                print(json.dumps({"ds": args.dataset, "train": row["train_dict"], "method": method,
                                  "F1": round(task_m["test_macro_f1"], 3), "AP_eval": round(m["ap"], 3),
                                  "mAP_eval": round(m["macro_ap"], 3), "R@P70": round(m["recall_at_p70"], 3)}),
                      flush=True)


if __name__ == "__main__":
    main()
