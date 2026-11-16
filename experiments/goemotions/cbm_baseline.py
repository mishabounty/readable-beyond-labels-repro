#!/usr/bin/env python
"""Быстрая проверка узкого места: joint CBM против нашего совместного обучения без bottleneck.

Joint CBM: z = normalize(MLP(x)) -> логиты концептов (линейно) -> sigmoid -> классификатор задачи ТОЛЬКО по
вероятностям концептов. Потеря: CE + lambda * BCE концептов (та же нормированная BCE, что у нас). Ранняя остановка
по val macro-F1 задачи. Метрики на test: macro-F1 задачи; AP концептов по собственным выходам CBM; AP свежего
декодера этапа 2 на z (как в train.py). Результат дописывается в results/cbm_<датасет>.jsonl.
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

HERE = Path(__file__).resolve().parent


class Args:
    epochs, patience, batch, lr, gate_epochs = 40, 6, 256, 1e-3, 30


def run(dataset, k, lam, seed, dictionary="ngram"):
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    x, y, split, M = T.load(dataset, dictionary, k, "mqa", False, seed)
    n_cls = int(y.max()) + 1
    enc = T.Encoder(x.shape[1]).to(T.DEV)
    concept = T.concept_head(768, M.shape[1]).to(T.DEV)
    task = nn.Linear(M.shape[1], n_cls).to(T.DEV)
    opt = torch.optim.AdamW([{"params": list(enc.parameters()) + list(task.parameters()), "lr": Args.lr},
                             {"params": concept.parameters(), "lr": 1e-2}], weight_decay=0.01)

    def predict(xb):
        logits = concept(enc(xb))
        return logits, task(torch.sigmoid(logits))

    best, stale = None, 0
    for ep in range(1, Args.epochs + 1):
        enc.train()
        for b in T.batches(split["fit"], Args.batch, True, gen):
            c_logits, t_logits = predict(x[b].to(T.DEV))
            loss = F.cross_entropy(t_logits, y[b].to(T.DEV)) + lam * T.bce_concepts(c_logits, T.dense(M, b).to(T.DEV))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        enc.eval()
        with torch.no_grad():
            pv = predict(x[split["val"]].to(T.DEV))[1].argmax(1).cpu()
        f1 = T.macro_f1(y[split["val"]], pv, n_cls)
        if best is None or f1 > best[0] + 1e-4:
            state = [{k_: v.detach().clone() for k_, v in m.state_dict().items()} for m in (enc, concept, task)]
            best, stale = (f1, ep, state), 0
        else:
            stale += 1
            if stale >= Args.patience:
                break
    best_epoch = best[1]
    for m, st in zip((enc, concept, task), best[2]):
        m.load_state_dict(st)
    enc.eval()
    with torch.no_grad():
        outs = [predict(x[i].to(T.DEV)) for i in torch.split(split["test"], 1024)]
        c_test = torch.cat([o[0].cpu() for o in outs])
        t_test = torch.cat([o[1].cpu() for o in outs])
        own = T.concept_metrics(c_test, T.dense(M, split["test"]))  # на CPU: при 5000 концептов не помещается в GPU
        z_all = torch.cat([enc(x[i:i + 4096].to(T.DEV)).cpu() for i in range(0, len(x), 4096)])
    yt = y[split["test"]]
    pt = t_test.argmax(1).cpu()
    del enc, concept, task, opt, outs
    torch.cuda.empty_cache()  # освобождаем GPU перед свежим декодером: при 5000 концептов иначе не хватает 4 ГБ
    probe = T.stage2(z_all, split, M, Args, gen)
    return {"dataset": dataset, "k": M.shape[1], "lambda": lam, "seed": seed, "best_epoch": best_epoch,
            "test_macro_f1": T.macro_f1(yt, pt, n_cls), "own_concepts": own, "probe_on_z": probe}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--settings", nargs="+", default=["agnews:1000", "agnews:5000", "20ng:1000", "goemotions:1000"])
    ap.add_argument("--lambdas", nargs="+", type=float, default=[1.0])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    for s in args.settings:
        parts = s.split(":")
        ds, k = parts[0], parts[1]
        dic = parts[2] if len(parts) > 2 else "ngram"
        for lam in args.lambdas:
            t0 = time.time()
            r = run(ds, int(k), lam, args.seed, dic)
            r["dict"] = dic
            r["seconds"] = time.time() - t0
            with open(HERE / "results" / f"cbm_{ds}.jsonl", "a") as fh:
                fh.write(json.dumps(r) + "\n")
            print(json.dumps({"ds": ds, "k": r["k"], "lam": lam, "F1": round(r["test_macro_f1"], 3),
                              "AP_own": round(r["own_concepts"]["ap"], 3),
                              "AP_probe": round(r["probe_on_z"]["ap"], 3), "ep": r["best_epoch"]}), flush=True)


if __name__ == "__main__":
    main()
