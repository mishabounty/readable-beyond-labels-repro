#!/usr/bin/env python
"""Пустышка CEM, зависящая от входа (план experiments/PLAN_2026-11-04_placebo3.md): метки уровня вида — случайные плоскости
в средних признаках исходной ResNet-34 с долями и примерно связями настоящих атрибутов. Зерно 4000 + повтор.
Настоящие метки используются только для долей и матрицы корреляций столбцов.
Результат: ~/cem_repro/results/cub_subsample_0.25/placebo3_data/fold_<k>/{train,val,test}.pkl и таблица свойств
placebo3_properties.json.
"""

import json
import os
import pickle
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore", category=ConvergenceWarning)
REPO = os.path.expanduser("~/cem_repro/cem")
BASE_DIR = os.path.join(REPO, "cem", "data", "CUB200", "class_attr_data_10")
RES = os.path.expanduser("~/cem_repro/results/cub_subsample_0.25")
OUT = os.path.join(RES, "placebo3_data")
MIN_POS = 5


def ap_ties(y, sc, rng, n=20):
    """AP при связках оценок — среднее по случайным разбиениям связок."""
    e = 1e-9 * (sc.std() + 1e-12)
    return float(np.mean([average_precision_score(y, sc + e * rng.randn(len(sc))) for _ in range(n)]))


def oof_ap(X, Y, seed, C=0.01):
    """Средняя AP столбцов Y вне выборки (5 частей по видам); столбцы с < MIN_POS положительных пропускаются."""
    rng, res = np.random.RandomState(seed), []
    folds = list(KFold(5, shuffle=True, random_state=seed).split(X))
    for j in range(Y.shape[1]):
        y = Y[:, j]
        if y.sum() < MIN_POS or y.sum() > len(y) - MIN_POS:
            continue
        sc = np.zeros(len(y))
        for tr, te in folds:
            if y[tr].min() == y[tr].max():
                sc[te] = y[tr].mean()
                continue
            mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
            sc[te] = LogisticRegression(C=C, max_iter=1000).fit((X[tr] - mu) / sd, y[tr]).decision_function((X[te] - mu) / sd)
        res.append(ap_ties(y, sc, rng))
    return float(np.mean(res))


def phi(a, b):
    """Корреляции столбцов a со столбцами b (постоянные столбцы дают 0)."""
    az = (a - a.mean(0)) / (a.std(0) + 1e-12)
    bz = (b - b.mean(0)) / (b.std(0) + 1e-12)
    return az.T @ bz / len(a)


def properties(tab, L, X, sel, held, seed):
    """Свойства 28 столбцов tab[:, sel] по видам."""
    A, c = tab[:, sel].astype(float), np.abs(phi(tab[:, sel].astype(float), tab[:, sel].astype(float)))
    return {"pos_rate": float(A.mean()), "distinct_sets": len({tuple(r) for r in A}),
            "mean_abs_phi": float(c[~np.eye(len(sel), dtype=bool)].mean()),
            "predict_from_appearance_AP": oof_ap(X, tab[:, sel], seed),
            "leak_AP_of_B": oof_ap(A, L[:, held], seed),
            "chance_B": float(L[:, held].mean()),
            "max_phi_with_real": float(np.abs(phi(A, L.astype(float))).max(1).mean()) if tab is not L else None}


def main():
    data = {s: pickle.load(open(os.path.join(BASE_DIR, f"{s}.pkl"), "rb")) for s in ("train", "val", "test")}
    table = {}
    for rows in data.values():
        for d in rows:
            table[d["class_label"]] = np.array(d["attribute_label"], dtype=np.int64)
    L = np.stack([table[c] for c in range(200)])
    pre = np.load(os.path.join(RES, "features", "pretrained_resnet34_imagenet_full.npz"))
    X = np.stack([pre["F_train"][pre["Y_train"].astype(int) == c].mean(0) for c in range(200)])
    X = X - X.mean(0)
    S = np.corrcoef(L.T)
    w, V = np.linalg.eigh((S + S.T) / 2)
    root = V @ np.diag(np.sqrt(np.clip(w, 1e-6, None)))  # root @ root.T ≈ S
    rows_out = {}
    for fold in range(5):
        rng = np.random.RandomState(4000 + fold)
        Z = X @ rng.randn(X.shape[1], L.shape[1]) @ root.T
        P = np.zeros_like(L)
        for j in range(L.shape[1]):
            P[np.argsort(-Z[:, j])[:int(L[:, j].sum())], j] = 1
        assert np.array_equal(P.sum(0), L.sum(0))
        d_out = os.path.join(OUT, f"fold_{fold}")
        os.makedirs(d_out, exist_ok=True)
        for s, rows in data.items():
            pickle.dump([{**d, "attribute_label": P[d["class_label"]].tolist()} for d in rows],
                        open(os.path.join(d_out, f"{s}.pkl"), "wb"))
        sel = np.load(os.path.join(RES, f"selected_concepts_sampling_0.25_fold_{fold}.npy"))
        held = np.setdiff1d(np.arange(112), sel)

        def other(name):
            tab = {}
            for d in pickle.load(open(os.path.join(RES, f"{name}_data", f"fold_{fold}", "train.pkl"), "rb")):
                tab[d["class_label"]] = np.array(d["attribute_label"], dtype=np.int64)
            return np.stack([tab[c] for c in range(200)])
        row = {"real": properties(L, L, X, sel, held, fold),
               "placebo": properties(other("placebo"), L, X, sel, held, fold),
               "placebo2": properties(other("placebo2"), L, X, sel, held, fold),
               "placebo3": properties(P, L, X, sel, held, fold)}
        row["agreement_with_real"] = float((P[:, sel] == L[:, sel]).mean())
        rows_out[fold] = row
        print(f"fold {fold}: совпадение placebo3 с настоящими {row['agreement_with_real']:.3f}", flush=True)
        for k in ("real", "placebo", "placebo2", "placebo3"):
            print(f"  {k:9s} " + ", ".join(f"{a} {v:.3f}" if isinstance(v, float) else f"{a} {v}" for a, v in row[k].items()), flush=True)
    json.dump(rows_out, open(os.path.join(RES, "placebo3_properties.json"), "w"), indent=1)
    print("PLACEBO3_DATA_DONE", flush=True)


if __name__ == "__main__":
    main()
