#!/usr/bin/env python
"""Сводка опытов Г1 (Hybrid как в статье) и Г2 (пустышка «случайный код») по плану experiments/PLAN_2026-11-10_checks.md.
Читает probe_results.jsonl (п1 — acc_c1, п2 — acc_tune, AP — ap_tune) и test_results_<модель>_fold_<k>.json.
Итог: stdout и results/cub_subsample_0.25/analysis_2026_11_10.json.
"""

import json
import os

import numpy as np
from scipy import stats

RES = os.path.expanduser("~/cem_repro/results/cub_subsample_0.25")
FOLDS = [0, 1, 2, 3, 4]
EQ = 0.3  # граница «неотличимы», п.


def interval(d):
    d = np.asarray(d, float)
    h = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    return float(d.mean()), float(d.mean() - h), float(d.mean() + h)


def main():
    rows = {}
    for s in open(os.path.join(RES, "probe_results.jsonl")):
        r = json.loads(s)
        rows[(r["model"], int(r["fold"]))] = r  # при повторе — последняя запись
    out = {}

    def pair(a, b, key, scale):
        if not all((m, f) in rows for m in (a, b) for f in FOLDS):
            return None
        d = [scale * (rows[(a, f)][key] - rows[(b, f)][key]) for f in FOLDS]
        m, lo, hi = interval(d)
        out[f"{a}-{b}|{key}"] = {"mean": m, "lo": lo, "hi": hi, "by_fold": d}
        return m, lo, hi, d

    names = {"acc_c1": "п1 (C = 1)", "acc_tune": "п2 (C по val)", "ap_tune": "AP (п2)"}
    print("=== Г1. Hybrid как в статье (hybrid448)")
    for a, b in (("cem", "hybrid448"), ("cem", "hybrid"), ("hybrid448", "hybrid")):
        for key, sc in (("acc_c1", 100.0), ("acc_tune", 100.0), ("ap_tune", 1.0)):
            r = pair(a, b, key, sc)
            if r:
                print(f"  {a:9s} − {b:9s} {names[key]:14s}: {r[0]:+.3f} [{r[1]:+.3f}; {r[2]:+.3f}]  по повторам "
                      f"{np.round(r[3], 3).tolist()}", flush=True)
    for m in ("cem", "hybrid", "hybrid448"):
        if all((m, f) in rows for f in FOLDS):
            print(f"  уровень {m:9s}: п1 {np.mean([100 * rows[(m, f)]['acc_c1'] for f in FOLDS]):.2f}, "
                  f"п2 {np.mean([100 * rows[(m, f)]['acc_tune'] for f in FOLDS]):.2f}", flush=True)
    g = out.get("cem-hybrid448|acc_c1")
    if g:
        d = np.array(g["by_fold"])
        se_our2 = d.var(ddof=1) / len(d)
        t4 = stats.t.ppf(0.975, 4)
        out["welch_448"] = {}
        for reading, conv in (("95% ДИ", lambda h: h / t4), ("SD", lambda h: h / np.sqrt(5)), ("SE", lambda h: h)):
            s1, s2 = conv(0.88) ** 2, conv(0.51) ** 2
            se = np.sqrt(se_our2 + s1 + s2)
            t = (d.mean() - 2.50) / se
            df = (se_our2 + s1 + s2) ** 2 / (se_our2 ** 2 / 4 + s1 ** 2 / 4 + s2 ** 2 / 4)
            p = float(2 * stats.t.sf(abs(t), df))
            out["welch_448"][reading] = {"t": float(t), "df": float(df), "p": p}
            print(f"  Уэлч против 2.50 ({reading}): t = {t:.2f}, df ≈ {df:.1f}, p = {p:.4f}", flush=True)
        out["rule_G1"] = {"direction_repeated": g["lo"] > 0, "magnitude_not_rejected_95CI": out["welch_448"]["95% ДИ"]["p"] >= 0.05}
        print(f"  ПРАВИЛО Г1: направление повторилось — {g['lo'] > 0}; 2.50 не отвергается (95% ДИ) — "
              f"{out['welch_448']['95% ДИ']['p'] >= 0.05}", flush=True)

    print("\n=== Г2. Пустышка «случайный код» (placebo4)")
    for a, b in (("placebo4", "cem"), ("placebo4", "placebo"), ("placebo4", "placebo2"), ("placebo4", "placebo3"),
                 ("placebo4", "cemnoc"), ("cem", "placebo")):
        for key, sc in (("acc_tune", 100.0), ("acc_c1", 100.0), ("ap_tune", 1.0)):
            r = pair(a, b, key, sc)
            if r:
                print(f"  {a:9s} − {b:9s} {names[key]:14s}: {r[0]:+.3f} [{r[1]:+.3f}; {r[2]:+.3f}]  по повторам "
                      f"{np.round(r[3], 3).tolist()}", flush=True)
    d2, d1 = out.get("placebo4-cem|acc_tune"), out.get("placebo4-cem|acc_c1")
    if d2 and d1:
        if d2["lo"] > 0 and d1["lo"] > 0:
            v = "бессмысленные метки читают B лучше настоящих"
        elif all(-EQ <= x["lo"] and x["hi"] <= EQ for x in (d1, d2)):
            v = f"неотличимы от настоящих в пределах {EQ} п."
        elif d2["hi"] < 0 and d1["hi"] < 0:
            v = "настоящие лучше"
        else:
            v = "не решено"
        out["rule_G2"] = v
        print(f"  ПРАВИЛО Г2: {v}", flush=True)
    print("\nточность понятий и вида на тесте (по test_results):", flush=True)
    for m in ("cem", "placebo", "placebo2", "placebo3", "placebo4", "cemnoc", "hybrid", "hybrid448"):
        c, y = [], []
        for f in FOLDS:
            p = os.path.join(RES, f"test_results_{m}_fold_{f}.json")
            if os.path.exists(p):
                t = json.load(open(p))
                c.append(t.get("test_c_accuracy")), y.append(t.get("test_y_accuracy"))
        if c:
            out[f"test|{m}"] = {"c_acc": c, "y_acc": y}
            print(f"  {m:9s}: понятия {np.mean([v for v in c if v is not None]):.4f}, вид "
                  f"{np.mean([v for v in y if v is not None]):.4f} ({len(c)} повторов)", flush=True)
    pp = os.path.join(RES, "placebo4_properties.json")
    if os.path.exists(pp):
        out["placebo4_properties"] = json.load(open(pp))
    json.dump(out, open(os.path.join(RES, "analysis_2026_11_10.json"), "w"), ensure_ascii=False, indent=1)
    print("G_DONE", flush=True)


if __name__ == "__main__":
    main()
