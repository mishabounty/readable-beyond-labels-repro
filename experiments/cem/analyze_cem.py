#!/usr/bin/env python
"""Анализ повторения CEM §5.3 / A.9 по плану, зафиксированному 28.10.2026 (HANDOFF §14.1).

Главная конечная точка: D = CEM − пустышка (обе пустышки, если есть), средняя точность пробы на 84 непройденных атрибутах (тест), парно по
повторам, 95% t-интервал. Заголовочное число — проба 2 (acc_tune), рядом проба 1 (acc_c1); вердикт только если обе
пробы дают один исход, иначе И0. Δ = 1.0 п.; исходы по L, U (границы интервала):
  И3 L > Δ; И3' 0 < L ≤ Δ ≤ U; И2 L > 0, U < Δ; И1 L ≤ 0, U < Δ; И0 L ≤ 0, U ≥ Δ.
Анализ также без разбиения 0 (видено до фиксации плана). AP — описательно.
Q1 (повторение Ф1): G = CEM − Hybrid по пробе 1, интервал; «направление повторилось» — L > 0; «величина» — Уэлч
против 2.50, у авторов «±» — 95% ДИ по 5 прогонам (SD = полуширина · √5 / t(0.975, 4)).
Запуск: python analyze_cem.py [путь к probe_results.jsonl]
"""

import json
import os
import sys

import numpy as np
from scipy import stats

DELTA = 1.0
F = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser(
    "~/cem_repro/results/cub_subsample_0.25/probe_results.jsonl")


def load():
    rows = {}
    for line in open(F):
        r = json.loads(line)
        rows[(r["model"], r["fold"])] = r  # при повторах берётся последняя строка
    return rows


def interval(d):
    d = np.asarray(d, dtype=float)
    n = len(d)
    m = d.mean()
    if n < 2:
        return m, np.nan, np.nan, np.nan
    sd = d.std(ddof=1)
    h = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)
    return m, m - h, m + h, sd


def outcome(lo, hi):
    if np.isnan(lo):
        return "—"
    if lo > DELTA:
        return "И3 смысл существенен"
    if lo > 0 and hi >= DELTA:
        return "И3' вклад есть, существенность не определена"
    if lo > 0:
        return "И2 малый, но настоящий вклад"
    if hi < DELTA:
        return "И1 вклад несущественен"
    return "И0 не решено"


def paired(rows, a, b, key, folds):
    fs = [f for f in folds if (a, f) in rows and (b, f) in rows]
    d = [100 * (rows[(a, f)][key] - rows[(b, f)][key]) if key != "ap_tune" else rows[(a, f)][key] - rows[(b, f)][key]
         for f in fs]
    return fs, d


def main():
    rows = load()
    folds = sorted({f for (_, f) in rows})
    models = ["pretrained_resnet34", "noconcept", "hybrid", "placebo", "placebo2", "cem"]
    print("по повторам (проба 1 / проба 2 / AP):")
    for f in folds:
        cells = []
        for m in models:
            r = rows.get((m, f))
            cells.append(f"{m} {100 * r['acc_c1']:.2f}/{100 * r['acc_tune']:.2f}/{r['ap_tune']:.3f}" if r else f"{m} —")
        print(f"  fold {f}: " + "; ".join(cells))
    for label, use in (("все повторы", folds), ("без разбиения 0", [f for f in folds if f != 0])):
        verdicts, uppers = [], []
        for b, bname in (("placebo", "пустышка-1"), ("placebo2", "пустышка-2")):
            fs0, _ = paired(rows, "cem", b, "acc_tune", use)
            if len(fs0) < 2:
                continue
            print(f"\n== D = CEM − {bname}, {label}")
            for key, name in (("acc_tune", "проба 2 (главная)"), ("acc_c1", "проба 1"), ("ap_tune", "AP, описательно")):
                fs, d = paired(rows, "cem", b, key, use)
                m, lo, hi, sd = interval(d)
                unit = "" if key == "ap_tune" else " п."
                oc = outcome(lo, hi) if key != "ap_tune" else ""
                if key != "ap_tune":
                    verdicts.append(oc.split()[0] if oc != "—" else "—")
                    uppers.append(hi)
                print(f"  {name}: n={len(fs)} разницы {np.round(d, 3).tolist()} среднее {m:+.3f}{unit} "
                      f"95% [{lo:+.3f}; {hi:+.3f}] SD {sd:.3f}  {oc}")
        # вердикт по плану §14.1: только если совпадают обе пробы и (когда есть) обе пустышки
        if verdicts:
            v = verdicts[0] if len(set(verdicts)) == 1 else "И0"
            print(f"  ВЕРДИКТ ({label}): {v}; исходы по пробам и пустышкам {verdicts}; "
                  f"наибольшая верхняя граница {max(uppers):+.3f} п. (Δ = {DELTA})")
    # Q1: повторение Ф1
    fs, g = paired(rows, "cem", "hybrid", "acc_c1", folds)
    if len(fs) >= 2:
        m, lo, hi, sd = interval(g)
        print(f"\n== Q1: G = CEM − Hybrid, проба 1: n={len(fs)} {np.round(g, 2).tolist()} среднее {m:+.2f} "
              f"95% [{lo:+.2f}; {hi:+.2f}]; направление {'повторилось' if lo > 0 else 'не установлено'}")
        t4 = stats.t.ppf(0.975, 4)
        sd_cem, sd_hyb = 0.88 * np.sqrt(5) / t4, 0.51 * np.sqrt(5) / t4
        se_their = np.sqrt(sd_cem ** 2 / 5 + sd_hyb ** 2 / 5)
        se_our = sd / np.sqrt(len(g))
        tw = (m - 2.50) / np.sqrt(se_our ** 2 + se_their ** 2)
        dfw = (se_our ** 2 + se_their ** 2) ** 2 / (se_our ** 4 / (len(g) - 1) + se_their ** 4 / 4)
        p = 2 * stats.t.sf(abs(tw), dfw)
        print(f"  Уэлч против 2.50 (их «±» как 95% ДИ): t = {tw:.2f}, df ≈ {dfw:.1f}, p = {p:.4f}")
        for m_ in ("cem", "hybrid"):
            v = [100 * rows[(m_, f)]["acc_c1"] for f in fs]
            print(f"  уровень {m_}: {np.mean(v):.2f} (у авторов {'94.33' if m_ == 'cem' else '91.83'})")


if __name__ == "__main__":
    main()
