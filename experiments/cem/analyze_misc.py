#!/usr/bin/env python
"""Прочие проверки по рецензиям 30.10 (план experiments/PLAN_2026-10-30_checks.md, пункты A5, A7, A8). Только CPU.

A5 — взаимодействие «модель × место»: по повторам (CEM_F − CEM_A) − (Hybrid_F − Hybrid_A) на обычной пробе и на
     непройденных видах (из analysis_extra.json), все 5 значений и без повтора 3; MLP-проба (скрытый слой 256,
     early stopping) на обычной пробе для CEM и Hybrid на узком месте F и на avgpool A.
A7 — Уэлч: наш разрыв CEM − Hybrid (проба 1) против опубликованных 2.50 при трёх прочтениях «±» авторов:
     95% ДИ по 5 прогонам, SD, SE.
A8 — главная конечная точка (CEM − пустышка-1/-2, пробы 1 и 2) без повторов 0 и 1 (видены до плана §14.1).
Итог: results/cub_subsample_0.25/analysis_misc.json и сводка в stdout.
Запуск: NJ=6 python analyze_misc.py
"""

import json
import os
import sys
import warnings

import numpy as np
from joblib import Parallel, delayed
from scipy import stats
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import average_precision_score
from sklearn.neural_network import MLPClassifier

sys.path.insert(0, os.path.expanduser("~/cem_repro"))
import run_subsample_2022 as R  # noqa: E402

FD = os.path.join(R.OUT, "features")
FOLDS = [0, 1, 2, 3, 4]
NJ = int(os.environ.get("NJ", "6"))
SMOKE = os.environ.get("SMOKE") == "1"  # пробный запуск: 4 атрибута в MLP-пробе


def interval(d):
    d = np.asarray(d, float)
    h = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    return [float(d.mean()), float(d.mean() - h), float(d.mean() + h)]


def mlp_probe(Ftr, Ctr, Fte, Cte, held):
    mu, sd = Ftr.mean(0), Ftr.std(0) + 1e-6
    tr, te = (Ftr - mu) / sd, (Fte - mu) / sd

    def one(j):
        ytr, yte = Ctr[:, j], Cte[:, j]
        if ytr.min() == ytr.max():
            return float((yte == ytr[0]).mean()), None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            m = MLPClassifier(hidden_layer_sizes=(256,), alpha=1e-4, early_stopping=True, max_iter=200,
                              random_state=0).fit(tr, ytr)
        acc = float((m.predict(te) == yte).mean())
        ap = float(average_precision_score(yte, m.predict_proba(te)[:, 1])) if yte.sum() >= 5 else None
        return acc, ap

    rows = Parallel(n_jobs=NJ)(delayed(one)(j) for j in held)
    return float(np.mean([r[0] for r in rows])), float(np.mean([r[1] for r in rows if r[1] is not None]))


def main():
    out = {}
    ex = json.load(open(os.path.join(R.OUT, "analysis_extra.json")))["rows"]
    # A5: взаимодействие «модель × место»
    for kind in ("std", "unseen"):
        vals = []
        for f in FOLDS:
            g = lambda m, k: ex[f"{m}|{k}|{kind}|{f}"][0]  # noqa: E731
            vals.append(100 * ((g("cem", "F") - g("cem", "A")) - (g("hybrid", "F") - g("hybrid", "A"))))
        out[f"interaction_{kind}"] = {"per_fold": vals, "all": interval(vals),
                                      "without_fold3": interval([v for f, v in zip(FOLDS, vals) if f != 3])}
        print(f"A5 взаимодействие ({kind}): по повторам {np.round(vals, 2).tolist()}; "
              f"все {np.round(out[f'interaction_{kind}']['all'], 2).tolist()}; "
              f"без повтора 3 {np.round(out[f'interaction_{kind}']['without_fold3'], 2).tolist()}", flush=True)
    # A5: MLP-проба на обычной пробе
    mlp = {}
    for f in FOLDS:
        held = np.setdiff1d(np.arange(112), R.selected_concepts(f))
        if SMOKE:
            held = held[:4]
        for m in ("cem", "hybrid"):
            z = np.load(os.path.join(FD, f"{m}_fold{f}_full.npz"))
            for k in ("F", "A"):
                mlp[f"{m}|{k}|{f}"] = mlp_probe(z[f"{k}_train"], z["C_train"].astype(int), z[f"{k}_test"],
                                                z["C_test"].astype(int), held)
                print(f"A5 MLP {m} {k} fold {f}: точность {100 * mlp[f'{m}|{k}|{f}'][0]:.2f}, "
                      f"AP {mlp[f'{m}|{k}|{f}'][1]:.4f}", flush=True)
    out["mlp"] = mlp
    comps = {"MLP Hybrid A − линейная CEM F": [100 * (mlp[f"hybrid|A|{f}"][0] - ex[f"cem|F|std|{f}"][0]) for f in FOLDS],
             "MLP Hybrid F − линейная CEM F": [100 * (mlp[f"hybrid|F|{f}"][0] - ex[f"cem|F|std|{f}"][0]) for f in FOLDS],
             "MLP CEM F − MLP Hybrid F": [100 * (mlp[f"cem|F|{f}"][0] - mlp[f"hybrid|F|{f}"][0]) for f in FOLDS],
             "MLP CEM A − MLP Hybrid A": [100 * (mlp[f"cem|A|{f}"][0] - mlp[f"hybrid|A|{f}"][0]) for f in FOLDS]}
    out["mlp_paired"] = {k: interval(v) for k, v in comps.items()}
    for k, v in out["mlp_paired"].items():
        print(f"A5 {k}: {v[0]:+.2f} [{v[1]:+.2f}; {v[2]:+.2f}] п.", flush=True)
    # A7 и A8 по probe_results.jsonl
    rows = {}
    for line in open(os.path.join(R.OUT, "probe_results.jsonl")):
        r = json.loads(line)
        rows[(r["model"], r["fold"])] = r
    g = [100 * (rows[("cem", f)]["acc_c1"] - rows[("hybrid", f)]["acc_c1"]) for f in FOLDS]
    m_, sd_ = float(np.mean(g)), float(np.std(g, ddof=1))
    se_our2 = sd_ ** 2 / len(g)
    t4 = stats.t.ppf(0.975, 4)
    out["welch"] = {}
    for reading, conv in (("95% ДИ", lambda h: h / t4), ("SD", lambda h: h / np.sqrt(5)), ("SE", lambda h: h)):
        s_cem2, s_hyb2 = conv(0.88) ** 2, conv(0.51) ** 2
        se = np.sqrt(se_our2 + s_cem2 + s_hyb2)
        t = (m_ - 2.50) / se
        df = (se_our2 + s_cem2 + s_hyb2) ** 2 / (se_our2 ** 2 / 4 + s_cem2 ** 2 / 4 + s_hyb2 ** 2 / 4)
        p = float(2 * stats.t.sf(abs(t), df))
        out["welch"][reading] = {"t": float(t), "df": float(df), "p": p}
        print(f"A7 Уэлч ({reading}): наш разрыв {m_:+.2f}, t = {t:.2f}, df ≈ {df:.1f}, p = {p:.4f}", flush=True)
    out["a8"] = {}
    for b in ("placebo", "placebo2"):
        for key, name in (("acc_tune", "проба 2"), ("acc_c1", "проба 1")):
            d = [100 * (rows[("cem", f)][key] - rows[(b, f)][key]) for f in (2, 3, 4)]
            out["a8"][f"{b}|{key}"] = interval(d)
            print(f"A8 CEM − {b}, {name}, повторы 2–4: {np.round(interval(d), 3).tolist()}", flush=True)
    json.dump(out, open(os.path.join(R.OUT, ("smoke_" if SMOKE else "") + "analysis_misc.json"), "w"),
              ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
