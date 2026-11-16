#!/usr/bin/env python
"""Избирательность меры A.9 (план experiments/PLAN_2026-11-11_amendments.md, П3; по образцу контрольных задач Hewitt и
Liang 2019): читаемость настоящих 84 атрибутов B минус читаемость 84 «контрольных» — тот же атрибут B, перемешанный
между видами (своя перестановка видов на атрибут, зерно 9200 + повтор): та же доля «да», уровень вида, но без связи с
внешностью. Проба — та же функция probe_heldout.probe (п1, п2, AP) на сохранённых признаках F.
Читаемость настоящих B берётся из probe_results.jsonl (та же функция); контрольная считается здесь.
Итог: results/cub_subsample_0.25/analysis_selectivity.json и сводка в stdout. Только CPU.
Запуск: python analyze_selectivity.py --models cem placebo placebo2 placebo3 placebo4 placebo4w cemnoc hybrid
"""

import argparse
import json
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.expanduser("~/cem_repro"))
import probe_heldout as P  # noqa: E402

R = P.R
FD = os.path.join(R.OUT, "features")
FOLDS = [0, 1, 2, 3, 4]


def interval(d):
    d = np.asarray(d, float)
    h = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    return [float(d.mean()), float(d.mean() - h), float(d.mean() + h)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    args = ap.parse_args()
    out_f = os.path.join(R.OUT, "analysis_selectivity.json")
    out = json.load(open(out_f)) if os.path.exists(out_f) else {"ctrl": {}}
    true = {}
    for s in open(os.path.join(R.OUT, "probe_results.jsonl")):
        r = json.loads(s)
        true[(r["model"], int(r["fold"]))] = r
    for fold in FOLDS:
        held = np.setdiff1d(np.arange(R.N_CONCEPTS), R.selected_concepts(fold))
        rng = np.random.RandomState(9200 + fold)
        perms = {int(j): rng.permutation(200) for j in held}  # одна и та же контрольная задача для всех моделей
        for m in args.models:
            key = f"{m}|{fold}"
            if key in out["ctrl"]:
                continue
            p = os.path.join(FD, f"{m}_fold{fold}_full.npz")
            if not os.path.exists(p):
                print("нет признаков", p, flush=True)
                continue
            z = np.load(p)
            F = {s: z[f"F_{s}"] for s in ("train", "val", "test")}
            Y = {s: z[f"Y_{s}"].astype(int) for s in ("train", "val", "test")}
            C = {s: z[f"C_{s}"].astype(int) for s in ("train", "val", "test")}
            T = np.zeros((200, C["train"].shape[1]), int)
            for s in ("train", "val", "test"):
                T[Y[s]] = C[s]  # метки уровня вида
            Cc = {s: C[s].copy() for s in C}
            for j, pi in perms.items():
                for s in Cc:
                    Cc[s][:, j] = T[pi[Y[s]], j]
            res, _ = P.probe(F, Cc, held)
            out["ctrl"][key] = res
            json.dump(out, open(out_f, "w"), indent=1)
            print(f"{m:10s} повтор {fold}: контроль п1 {100 * res['acc_c1']:.2f}, п2 {100 * res['acc_tune']:.2f}, "
                  f"AP {res['ap_tune']:.4f}", flush=True)
    print("\nизбирательность = настоящие B − контрольные (п.; AP); разница с cem парно по повторам")
    summ = {}
    for m in args.models:
        fs = [f for f in FOLDS if f"{m}|{f}" in out["ctrl"] and (m, f) in true]
        if len(fs) < 2:
            continue
        for k, sc in (("acc_tune", 100), ("acc_c1", 100), ("ap_tune", 1)):
            summ[f"{m}|sel|{k}"] = [sc * (true[(m, f)][k] - out["ctrl"][f"{m}|{f}"][k]) for f in fs]
            summ[f"{m}|ctrl|{k}"] = [sc * out["ctrl"][f"{m}|{f}"][k] for f in fs]
        print(f"  {m:10s} избирательность п2 {interval(summ[f'{m}|sel|acc_tune'])[0]:+.2f}, п1 "
              f"{interval(summ[f'{m}|sel|acc_c1'])[0]:+.2f}, AP {interval(summ[f'{m}|sel|ap_tune'])[0]:+.4f}; "
              f"контроль п2 {interval(summ[f'{m}|ctrl|acc_tune'])[0]:.2f}", flush=True)
    for m in args.models:
        if m == "cem" or f"{m}|sel|acc_tune" not in summ or "cem|sel|acc_tune" not in summ:
            continue
        line = []
        for k in ("acc_tune", "acc_c1", "ap_tune"):
            d = np.array(summ[f"cem|sel|{k}"]) - np.array(summ[f"{m}|sel|{k}"])
            i = interval(d)
            out.setdefault("pairs", {})[f"cem-{m}|sel|{k}"] = i
            line.append(f"{k} {i[0]:+.3f} [{i[1]:+.3f}; {i[2]:+.3f}]")
        print(f"  cem − {m:10s} по избирательности: " + "; ".join(line), flush=True)
    out["summary"] = summ
    json.dump(out, open(out_f, "w"), indent=1)
    print("SELECTIVITY_DONE", flush=True)


if __name__ == "__main__":
    main()
