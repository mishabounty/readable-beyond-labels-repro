#!/usr/bin/env python
"""MLP-проба с подбором настроек (план experiments/PLAN_2026-11-02_checks.md, N6). Только CPU.

Обычная проба (видимые виды): обучение на фото train, подбор настроек по средней AP на фото val, проверка на test.
Модели CEM и Hybrid, места чтения: узкое место F и avgpool A. Настройки: alpha ∈ {1e-4, 1e-3, 1e-2, 0.1} × ширина
скрытого слоя ∈ {128, 256, 512}, early stopping, одна выбранная пара на модель × место × повтор. Для каждой пары
каждая сеть обучается один раз; итог на test берётся у пары, лучшей на val.
Сравнения по повторам (точность, п., и AP): MLP CEM F − MLP Hybrid F; MLP CEM A − MLP Hybrid A; взаимодействие
(CEM F − CEM A) − (Hybrid F − Hybrid A) при MLP; MLP Hybrid F − линейная CEM F (из analysis_extra.json). Все 5 значений
и без повтора 3.
Итог: results/cub_subsample_0.25/mlp_tuned.json. Запуск: NJ=12 python mlp_tuned.py; SMOKE=1 — пробный.
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
NJ = int(os.environ.get("NJ", "12"))
GRID = [(a, w) for a in (1e-4, 1e-3, 1e-2, 0.1) for w in (128, 256, 512)]
SMOKE = os.environ.get("SMOKE") == "1"
if SMOKE:
    FOLDS, GRID = [0, 1], [(1e-4, 128), (1e-2, 256)]


def interval(d):
    d = np.asarray(d, float)
    h = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    return [float(d.mean()), float(d.mean() - h), float(d.mean() + h)]


def one(tr, ytr, va, yva, te, yte, alpha, width):
    if ytr.min() == ytr.max():
        return float((yva == ytr[0]).mean()), None, float((yte == ytr[0]).mean()), None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        m = MLPClassifier(hidden_layer_sizes=(width,), alpha=alpha, early_stopping=True, max_iter=300,
                          random_state=0).fit(tr, ytr)
    pv, pt = m.predict_proba(va)[:, 1], m.predict_proba(te)[:, 1]
    apv = float(average_precision_score(yva, pv)) if yva.sum() >= 5 else None
    apt = float(average_precision_score(yte, pt)) if yte.sum() >= 5 else None
    return float(((pv > 0.5) == yva).mean()), apv, float(((pt > 0.5) == yte).mean()), apt


def main():
    out = {"grid": GRID, "rows": {}, "chosen": {}}
    ex = json.load(open(os.path.join(R.OUT, "analysis_extra.json")))["rows"]
    for f in FOLDS:
        held = np.setdiff1d(np.arange(112), R.selected_concepts(f))
        if SMOKE:
            held = held[:4]
        for m in ("cem", "hybrid"):
            z = np.load(os.path.join(FD, f"{m}_fold{f}_full.npz"))
            C = {s: z[f"C_{s}"].astype(int) for s in ("train", "val", "test")}
            for k in ("F", "A"):
                X = {s: z[f"{k}_{s}"] for s in ("train", "val", "test")}
                mu, sd = X["train"].mean(0), X["train"].std(0) + 1e-6
                X = {s: (v - mu) / sd for s, v in X.items()}
                tasks = [(g, j) for g in GRID for j in held]
                res = Parallel(n_jobs=NJ)(delayed(one)(X["train"], C["train"][:, j], X["val"], C["val"][:, j],
                                                       X["test"], C["test"][:, j], *g) for g, j in tasks)
                by = {g: [] for g in GRID}
                for (g, j), r in zip(tasks, res):
                    by[g].append(r)
                score = {g: np.mean([r[1] for r in by[g] if r[1] is not None]) for g in GRID}
                g = max(GRID, key=lambda x: score[x])
                acc = float(np.mean([r[2] for r in by[g]]))
                ap = float(np.mean([r[3] for r in by[g] if r[3] is not None]))
                out["rows"][f"{m}|{k}|{f}"] = [acc, ap]
                out["chosen"][f"{m}|{k}|{f}"] = {"alpha": g[0], "width": g[1], "val_ap": float(score[g])}
                print(f"MLP {m} {k} fold {f}: выбрано alpha {g[0]:g}, ширина {g[1]}; точность {100 * acc:.2f}, "
                      f"AP {ap:.4f}", flush=True)
    rows = out["rows"]
    g = lambda m, k, f, i=0: rows[f"{m}|{k}|{f}"][i]  # noqa: E731
    comps = {
        "MLP CEM F − MLP Hybrid F": lambda f, i: g("cem", "F", f, i) - g("hybrid", "F", f, i),
        "MLP CEM A − MLP Hybrid A": lambda f, i: g("cem", "A", f, i) - g("hybrid", "A", f, i),
        "взаимодействие при MLP": lambda f, i: (g("cem", "F", f, i) - g("cem", "A", f, i)) - (g("hybrid", "F", f, i) - g("hybrid", "A", f, i)),
        "MLP Hybrid F − линейная CEM F": lambda f, i: g("hybrid", "F", f, i) - ex[f"cem|F|std|{f}"][i],
    }
    out["paired"] = {}
    for name, fn in comps.items():
        for i, unit in ((0, "точность, п."), (1, "AP")):
            if name.endswith("линейная CEM F") and i == 1 and len(ex[f"cem|F|std|{FOLDS[0]}"]) < 2:
                continue
            sc = 100 if i == 0 else 1
            d = [sc * fn(f, i) for f in FOLDS]
            res = {"per_fold": d, "all": interval(d)}
            if 3 in FOLDS:
                res["without_fold3"] = interval([v for f, v in zip(FOLDS, d) if f != 3])
            out["paired"][f"{name}|{unit}"] = res
            print(f"{name} ({unit}): по повторам {np.round(d, 3).tolist()}; все {np.round(res['all'], 3).tolist()}"
                  + (f"; без повтора 3 {np.round(res['without_fold3'], 3).tolist()}" if "without_fold3" in res else ""),
                  flush=True)
    json.dump(out, open(os.path.join(R.OUT, ("smoke_" if SMOKE else "") + "mlp_tuned.json"), "w"), ensure_ascii=False, indent=1)
    print("MLP_DONE", flush=True)


if __name__ == "__main__":
    main()
