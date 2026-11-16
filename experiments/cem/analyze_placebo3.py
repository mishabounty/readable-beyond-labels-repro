#!/usr/bin/env python
"""Сводка по пустышке placebo3 (план experiments/PLAN_2026-11-04_placebo3.md): разницы обычной пробы, дерево §14.1 для
D_H = cem − placebo3, точность понятий и вида. Итог: stdout и results/.../analysis_placebo3.json.
"""

import json
import os

import numpy as np
from scipy import stats

RES = os.path.expanduser("~/cem_repro/results/cub_subsample_0.25")
FOLDS = [0, 1, 2, 3, 4]
DELTA = 1.0


def interval(d):
    d = np.asarray(d, float)
    h = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    return float(d.mean()), float(d.mean() - h), float(d.mean() + h)


def verdict(lo, hi):
    """Дерево исходов §14.1 (L, U — границы, Δ = 1 п.)."""
    if lo > DELTA:
        return "И3"
    if lo > 0:
        return "И3′" if hi >= DELTA else "И2"
    return "И1" if hi < DELTA else "И0"


def main():
    rows = {}
    for s in open(os.path.join(RES, "probe_results.jsonl")):
        r = json.loads(s)
        rows[(r["model"], int(r["fold"]))] = r  # при повторе берётся последняя запись
    out = {}
    pairs = [("cem", "placebo3"), ("placebo3", "placebo2"), ("placebo3", "placebo"), ("placebo3", "cemnoc"),
             ("placebo3", "hybrid"), ("cem", "placebo2"), ("cem", "cemnoc")]
    for key, scale in (("acc_tune", 100.0), ("acc_c1", 100.0), ("ap_tune", 1.0)):
        print(f"{key}:", flush=True)
        for a, b in pairs:
            if not all((m, f) in rows for m in (a, b) for f in FOLDS):
                print(f"  {a} − {b}: нет данных", flush=True)
                continue
            d = [scale * (rows[(a, f)][key] - rows[(b, f)][key]) for f in FOLDS]
            m, lo, hi = interval(d)
            out[f"{a}-{b}|{key}"] = {"mean": m, "lo": lo, "hi": hi, "by_fold": d}
            print(f"  {a:9s} − {b:9s}: {m:+.4f} [{lo:+.4f}; {hi:+.4f}]  по повторам {np.round(d, 4).tolist()}", flush=True)
    ver = {k: verdict(out[f"cem-placebo3|{k}"]["lo"], out[f"cem-placebo3|{k}"]["hi"]) for k in ("acc_tune", "acc_c1")}
    out["verdict"] = ver
    print(f"дерево §14.1 для D_H: {ver}; итог {ver['acc_tune'] if len(set(ver.values())) == 1 else 'не решено (И0)'}", flush=True)
    print("точность понятий и вида на тесте:", flush=True)
    for m in ("cem", "placebo", "placebo2", "placebo3"):
        c, y = [], []
        for f in FOLDS:
            p = os.path.join(RES, f"test_results_{m}_fold_{f}.json")
            if os.path.exists(p):
                d = json.load(open(p))
                c.append(d["test_c_accuracy"]), y.append(d["test_y_accuracy"])
        if c:
            print(f"  {m:9s}: понятия {100 * np.mean(c):.2f}, вид {100 * np.mean(y):.2f} (n = {len(c)})", flush=True)
            out[f"test|{m}"] = {"c_acc": c, "y_acc": y}
    json.dump(out, open(os.path.join(RES, "analysis_placebo3.json"), "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
