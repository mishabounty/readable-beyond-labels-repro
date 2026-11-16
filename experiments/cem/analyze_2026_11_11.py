#!/usr/bin/env python
"""Сводка П2 (placebo4w — «случайный код» с весами потерь настоящих меток) и П4 («случайный код» − hybrid448) по плану
experiments/PLAN_2026-11-11_amendments.md. Читает probe_results.jsonl и test_results_*. Итог — stdout и
results/cub_subsample_0.25/analysis_2026_11_11.json."""

import json
import os

import numpy as np
from scipy import stats

RES = os.path.expanduser("~/cem_repro/results/cub_subsample_0.25")
FOLDS = [0, 1, 2, 3, 4]


def interval(d):
    d = np.asarray(d, float)
    h = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    return [float(d.mean()), float(d.mean() - h), float(d.mean() + h)]


rows = {}
for s in open(os.path.join(RES, "probe_results.jsonl")):
    r = json.loads(s)
    rows[(r["model"], int(r["fold"]))] = r
out = {}
for a, b in (("placebo4w", "cem"), ("placebo4w", "placebo4"), ("placebo4", "hybrid448"), ("placebo4w", "hybrid448"),
             ("placebo4w", "placebo"), ("placebo4w", "cemnoc")):
    if not all((m, f) in rows for m in (a, b) for f in FOLDS):
        print(f"{a} − {b}: нет данных", flush=True)
        continue
    for k, sc in (("acc_tune", 100), ("acc_c1", 100), ("ap_tune", 1)):
        d = [sc * (rows[(a, f)][k] - rows[(b, f)][k]) for f in FOLDS]
        out[f"{a}-{b}|{k}"] = interval(d) + [d]
        i = out[f"{a}-{b}|{k}"]
        print(f"{a:9s} − {b:9s} {k:8s}: {i[0]:+.3f} [{i[1]:+.3f}; {i[2]:+.3f}]  по повторам {np.round(d, 3).tolist()}",
              flush=True)
d2, d1 = out.get("placebo4w-cem|acc_tune"), out.get("placebo4w-cem|acc_c1")
if d2 and d1:
    v = ("бессмысленные метки читают B лучше настоящих и при равном весе потерь" if d2[1] > 0 and d1[1] > 0 else
         ("преимущество Г2 объяснялось весом потерь" if d2[2] < 0 and d1[2] < 0 else "не решено"))
    out["rule_P2"] = v
    print("ПРАВИЛО П2:", v, flush=True)
for m in ("cem", "placebo4", "placebo4w"):
    y = [json.load(open(os.path.join(RES, f"test_results_{m}_fold_{f}.json"))).get("test_y_accuracy") for f in FOLDS
         if os.path.exists(os.path.join(RES, f"test_results_{m}_fold_{f}.json"))]
    c = [json.load(open(os.path.join(RES, f"test_results_{m}_fold_{f}.json"))).get("test_c_accuracy") for f in FOLDS
         if os.path.exists(os.path.join(RES, f"test_results_{m}_fold_{f}.json"))]
    if y:
        print(f"{m:9s}: точность вида {np.mean(y):.4f}, понятий {np.mean(c):.4f} ({len(y)} повторов)", flush=True)
json.dump(out, open(os.path.join(RES, "analysis_2026_11_11.json"), "w"), ensure_ascii=False, indent=1)
print("P2_DONE", flush=True)
