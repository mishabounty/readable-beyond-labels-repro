#!/usr/bin/env python
"""Проверки по рецензиям на v8 (план experiments/PLAN_2026-11-02_checks.md, пункты N1, N2, N4, N5, N8). Только CPU.

Проба на непройденных видах: учимся на фото train одной половины видов, проверяем на фото test другой.
  N1 — вложенная проба «сверх A»: вход [F, A_true], после стандартизации блок A умножается на s (штраф на A слабее
       в s² раз); (C, s) подбираются вместе по средней AP в GroupKFold (3 части по видам) на обучающей половине
       первого деления. Годность: AP[F, A] ≥ AP[A] − 0.002. Вторая проба STK: логит B по одним A_true (на обучающих
       фото — вне выборки, GroupKFold по видам) одним столбцом ×30 рядом с F. Мера Δ(F | A) = AP[F, A] − AP[A].
  Добавлено 02.11 после пробного прогона (SMOKE), до полного счёта: на фото, где училась модель, F CEM почти
  повторяет A, поэтому пробы FA и STK, обученные на этих фото, проигрывают одним A по построению (сдвиг «свои / чужие
  фото»). Две пробы без этого сдвига:
    OFF — двухшаговая: логит B по одним A_true (на обучающих фото — вне выборки) входит смещением с коэффициентом 1,
          поверх него обучается только F (L2 на F, C как у F); F может лишь добавить к A;
    FAu — вложенная проба, обученная на test-фото обучающей половины видов (модель их не видела), проверка — на test-фото
          проверочной половины; база — одни A на тех же фото (Au). Δ_u(F | A) = AP[FAu] − AP[Au].
  N2 (--half) — ориентиры опыта C: исходная ResNet на делениях C; читаемость самих A по F на S2 (вход FtoA);
       обычная проба на своих видах S1 train → S1 test (вход Fseen); проба, обученная на test-фото S1, которых модель
       не видела (вход Funs).
  N4 — порог положительных в видах (≥ 5 положительных видов в каждой половине); двухуровневый бутстрэп
       (повторы × проверочные виды первого деления, 2000 выборок, одна выборка для всех моделей) для AP сравнений.
  N5 — A6 исправлено: центрированная доля «да», распределение max|φ|, разница в нижней четверти по max|φ|.
  N8 (--half) — мощность: к F CEM без понятий добавляются столбцы настоящих B с шумом σ·sd; Δ(F′ | A) при каждом σ.
Без --half: модели видели все 200 видов (cem, placebo, placebo2, cemnoc, hybrid, noconcept, pretrained), 10 делений.
С --half: cem_half, placebo2_half, cemnoc_half, cemall_half (если есть), pretrained; деление S1/S2 = RandomState(100 + fold).
Итог: results/cub_subsample_0.25/analysis_nested[_half].json и сводка в stdout.
Запуск: NJ=12 python analyze_nested.py [--half]; SMOKE=1 — пробный (2 повтора, 1 деление, 6 атрибутов, узкие сетки).
"""

import argparse
import json
import os
import pickle
import sys
import warnings

import numpy as np
from joblib import Parallel, delayed
from scipy import stats
from scipy.optimize import minimize
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupKFold

sys.path.insert(0, os.path.expanduser("~/cem_repro"))
import run_subsample_2022 as R  # noqa: E402

warnings.filterwarnings("ignore", category=ConvergenceWarning)
FD = os.path.join(R.OUT, "features")
NJ = int(os.environ.get("NJ", "12"))
FOLDS = [0, 1, 2, 3, 4]
CS = [1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0]
SS = [1.0, 3.0, 10.0, 30.0, 100.0]
# Отступления от плана (после пробных прогонов, до полного счёта; записаны в CHECKS_2026-11-02.md):
#  2) масштаб блока A (s) заменён раздельными штрафами: логистическая регрессия с L2 |w_F|²/(2 C_F) + |w_A|²/(2 C_A)
#     (fit_two; записана через масштаб √C столбцов при единичном штрафе). Диагностика: при s ≈ 3000 решатель sklearn не сходится, и [F, A] с почти нулевым F давал 0.596 вместо
#     0.613 у одних A — вариант «опереться на A» был в сетке лишь формально. Сетка C_F × C_A (24 пары), подбор как у C;
#  3) у OFF свой C по GroupKFold по видам (CS_OFF), а не C пробы по одному F;
#  4) AP при совпадающих оценках (ap_score): у пробы по одним A оценка одинакова у всех фото вида и у видов с одинаковым
#     набором A (96 разных значений на 2896 фото), а average_precision_score считает связку одним порогом — это даёт
#     0.665 против 0.649 при случайном разбиении связок (диагностика 02.11). Теперь при связках AP — среднее по 20
#     случайным разбиениям (ожидание при случайном порядке внутри связки); для непрерывных оценок ничего не меняется.
CF_GRID = [1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0]
CA_GRID = [1e-3, 1e-2, 0.1, 1.0]
CS_A = [1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0]
CS_OFF = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0]
N_SPLITS = 10
N_BOOT = 2000
MIN_SP = 5  # положительных видов в каждой половине
SIGMAS = [1.0, 2.0, 4.0, 8.0, 16.0]
STK_SCALE = 30.0
N_A = 28
SMOKE = os.environ.get("SMOKE") == "1"
if SMOKE:
    FOLDS, N_SPLITS, CS, SS, N_BOOT, SIGMAS = [0, 1], 1, [1e-3, 0.1], [1.0, 100.0], 50, [2.0]
    CF_GRID, CA_GRID, CS_OFF = [1e-5, 1e-2], [1e-2, 0.1], [1e-6, 1e-3, 0.1]


def ap_score(y, sc, n_rep=20):
    """AP; при связках оценок — среднее по n_rep случайным разбиениям связок (детерминированно)."""
    y = np.asarray(y)
    if len(np.unique(sc)) > 0.98 * len(sc):
        return float(average_precision_score(y, sc))
    r = np.random.RandomState(12345)
    eps = 1e-9 * (np.std(sc) + 1e-12)
    return float(np.mean([average_precision_score(y, sc + eps * r.randn(len(sc))) for _ in range(n_rep)]))


def interval(d):
    d = np.asarray(d, float)
    if len(d) < 2:
        return [float(np.mean(d)) if len(d) else float("nan"), float("nan"), float("nan")]
    h = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    return [float(d.mean()), float(d.mean() - h), float(d.mean() + h)]


def species_table():
    L = {}
    for d in pickle.load(open(os.path.join(R.BASE_DIR, "train.pkl"), "rb")):
        L[d["class_label"]] = np.array(d["attribute_label"])
    return np.stack([L[k] for k in range(200)]).astype(float)  # 200 × 112, уровень вида


def prep(Xtr, Xte, s=1.0, n_a=0):
    """Стандартизация по обучающей части; последние n_a столбцов (блок A) умножаются на s."""
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    a, b = (Xtr - mu) / sd, (Xte - mu) / sd
    if n_a and s != 1.0:
        a[:, -n_a:] *= s
        b[:, -n_a:] *= s
    return a, b


def fit_scores(Xtr, ytr, Xte, C):
    """Оценки (decision_function) на проверочных фото; при одном классе в обучении — константа."""
    if ytr.min() == ytr.max():
        return np.full(len(Xte), float(ytr[0]))
    return LogisticRegression(C=C, max_iter=2000).fit(Xtr, ytr).decision_function(Xte)


def tune(X, Y, groups, n_a=0, scales=(1.0,), cs=None):
    """(C, s) по средней AP в GroupKFold по видам (3 части) на обучающей половине."""
    folds = list(GroupKFold(3).split(X, groups=groups))
    grid = [(C, s) for C in (cs or CS) for s in scales]

    def score(C, s, k, j):
        tr, va = folds[k]
        ytr, yva = Y[tr, j], Y[va, j]
        if ytr.min() == ytr.max() or yva.sum() < 1 or yva.sum() == len(yva):
            return None
        a, b = prep(X[tr], X[va], s, n_a)
        return ap_score(yva, fit_scores(a, ytr, b, C))

    tasks = [(C, s, k, j) for (C, s) in grid for k in range(3) for j in range(Y.shape[1])]
    res = Parallel(n_jobs=NJ)(delayed(score)(*t) for t in tasks)
    by = {g: [] for g in grid}
    for (C, s, _, _), r in zip(tasks, res):
        if r is not None:
            by[(C, s)].append(r)
    best = max(grid, key=lambda g: (np.mean(by[g]) if by[g] else -1, -abs(np.log10(g[0])), -g[1]))
    return best, {f"{C:g}|{s:g}": (float(np.mean(by[(C, s)])) if by[(C, s)] else None) for (C, s) in grid}


def eval_attrs(Xtr, Ytr, Xte, Yte, C, s=1.0, n_a=0, keep=None):
    """Оценки по всем атрибутам и AP по тем, что в keep (иначе NaN)."""
    a, b = prep(Xtr, Xte, s, n_a)
    sc = Parallel(n_jobs=NJ)(delayed(fit_scores)(a, Ytr[:, j], b, C) for j in range(Ytr.shape[1]))
    sc = np.stack(sc, 1)
    ap = np.array([ap_score(Yte[:, j], sc[:, j]) if (keep is None or keep[j]) and 0 < Yte[:, j].sum()
                   else np.nan for j in range(Yte.shape[1])])
    return sc, ap


def fit_offset(X, y, off, Xte, offte, C):
    """Логистическая регрессия со смещением off (коэффициент 1): минимизирует сумму логистических потерь + |w|² / (2C)."""
    if y.min() == y.max():
        return offte.copy()
    sgn = 2.0 * y - 1.0

    def f(p):
        w, b = p[:-1], p[-1]
        m = -sgn * (X @ w + b + off)
        sig = 0.5 * (1.0 + np.tanh(0.5 * m))  # сигмоида m без переполнения
        g = -sgn * sig
        return np.logaddexp(0, m).sum() + 0.5 * (w @ w) / C, np.append(X.T @ g + w / C, g.sum())
    r = minimize(f, np.zeros(X.shape[1] + 1), jac=True, method="L-BFGS-B", options={"maxiter": 500})
    return Xte @ r.x[:-1] + r.x[-1] + offte


def tune_off(Fstd, Y, off, groups):
    """C для OFF по средней AP в GroupKFold по видам; смещение — логит по A вне выборки (общий для внутренних частей)."""
    folds = list(GroupKFold(3).split(Fstd, groups=groups))

    def score(C, k, j):
        tr, va = folds[k]
        if Y[tr, j].min() == Y[tr, j].max() or Y[va, j].sum() < 1:
            return None
        sc = fit_offset(Fstd[tr], Y[tr, j].astype(float), off[tr, j], Fstd[va], off[va, j], C)
        return ap_score(Y[va, j], sc)
    tasks = [(C, k, j) for C in CS_OFF for k in range(3) for j in range(Y.shape[1])]
    res = Parallel(n_jobs=NJ)(delayed(score)(*t) for t in tasks)
    by = {C: [] for C in CS_OFF}
    for (C, _, _), r in zip(tasks, res):
        if r is not None:
            by[C].append(r)
    return max(CS_OFF, key=lambda c: (np.mean(by[c]) if by[c] else -1, -c))


def fit_two(Ftr, Atr, y, Fte, Ate, CF, CA):
    """Логистическая регрессия на [F, A] с раздельными штрафами |w_F|²/(2 CF) + |w_A|²/(2 CA); оценки на проверке.
    Равносильная запись: столбцы F умножаются на √CF, столбцы A — на √CA, штраф единичный (C = 1); так задача хорошо
    обусловлена при любом CF (при прямой записи с CF ≈ 1e-9 решатель останавливался рано)."""
    if y.min() == y.max():
        return np.full(len(Fte), float(y[0]))
    sf, sa = np.sqrt(CF), np.sqrt(CA)
    m = LogisticRegression(C=1.0, max_iter=3000).fit(np.hstack([sf * Ftr, sa * Atr]), y)
    return m.decision_function(np.hstack([sf * Fte, sa * Ate]))


def std2(Ftr, Fte, Atr, Ate):
    """Стандартизация блоков F и A по обучающей части (каждый столбец)."""
    a, b = prep(Ftr, Fte)
    c, d = prep(Atr, Ate)
    return a, b, c, d


def tune_two(F, A, Y, groups):
    """(C_F, C_A) по средней AP в GroupKFold по видам (3 части)."""
    folds = list(GroupKFold(3).split(F, groups=groups))
    parts = [std2(F[tr], F[va], A[tr], A[va]) for tr, va in folds]
    grid = [(cf, ca) for cf in CF_GRID for ca in CA_GRID]

    def score(cf, ca, k, j):
        tr, va = folds[k]
        if Y[tr, j].min() == Y[tr, j].max() or Y[va, j].sum() < 1:
            return None
        a, b, c, d = parts[k]
        return ap_score(Y[va, j], fit_two(a, c, Y[tr, j].astype(float), b, d, cf, ca))
    tasks = [(cf, ca, k, j) for (cf, ca) in grid for k in range(3) for j in range(Y.shape[1])]
    res = Parallel(n_jobs=NJ)(delayed(score)(*t) for t in tasks)
    by = {g: [] for g in grid}
    for (cf, ca, _, _), r in zip(tasks, res):
        if r is not None:
            by[(cf, ca)].append(r)
    best = max(grid, key=lambda g: (np.mean(by[g]) if by[g] else -1, -g[0]))
    return best, {f"{cf:g}|{ca:g}": (float(np.mean(by[(cf, ca)])) if by[(cf, ca)] else None) for (cf, ca) in grid}


def eval_two(Ftr, Atr, Ytr, Fte, Ate, Yte, cf, ca, keep):
    a, b, c, d = std2(Ftr, Fte, Atr, Ate)
    sc = np.stack(Parallel(n_jobs=NJ)(delayed(fit_two)(a, c, Ytr[:, j].astype(float), b, d, cf, ca)
                                      for j in range(Ytr.shape[1])), 1)
    ap = np.array([ap_score(Yte[:, j], sc[:, j]) if keep[j] and 0 < Yte[:, j].sum() else np.nan
                   for j in range(Yte.shape[1])])
    return sc, ap


def stack_logits(Atr, Btr, groups, Ate, C):
    """Логит B по одним A_true: на обучающих фото — вне выборки (GroupKFold по видам), на проверочных — по всей половине."""
    tr_l = np.zeros(Btr.shape, float)
    for a_i, b_i in GroupKFold(3).split(Atr, groups=groups):
        p, q = prep(Atr[a_i], Atr[b_i])
        cols = Parallel(n_jobs=NJ)(delayed(fit_scores)(p, Btr[a_i, j], q, C) for j in range(Btr.shape[1]))
        tr_l[b_i] = np.stack(cols, 1)
    p, q = prep(Atr, Ate)
    te_l = np.stack(Parallel(n_jobs=NJ)(delayed(fit_scores)(p, Btr[:, j], q, C) for j in range(Btr.shape[1])), 1)
    return tr_l, te_l


def weighted_ap(order, y, w):
    """AP при весах фото (кратности бутстрэпа); order — порядок по убыванию оценки."""
    yo, wo = y[order], w[order]
    tp = np.cumsum(wo * yo)
    tot = np.cumsum(wo)
    pos = (wo * yo).sum()
    return float((wo * yo * tp / np.maximum(tot, 1e-12)).sum() / pos) if pos > 0 else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--half", action="store_true")
    ap.add_argument("--models", nargs="*", help="свой список моделей вместо стандартного (без --half)")
    ap.add_argument("--pairs", nargs="*", help="свои сравнения вида a:b")
    ap.add_argument("--tag", default="", help="суффикс имени файла итогов")
    args = ap.parse_args()
    if args.half:
        models = ["cem_half", "placebo2_half", "cemnoc_half", "cemall_half", "pretrained"]
        pairs = [("cem_half", "placebo2_half"), ("cem_half", "cemnoc_half"), ("placebo2_half", "cemnoc_half"),
                 ("cemall_half", "cemnoc_half"), ("cemnoc_half", "pretrained"), ("cem_half", "pretrained")]
    else:
        models = ["cem", "placebo", "placebo2", "cemnoc", "hybrid", "noconcept", "pretrained"]
        pairs = [("cem", "placebo2"), ("cem", "placebo"), ("cem", "cemnoc"), ("placebo2", "cemnoc"),
                 ("cemnoc", "noconcept"), ("cem", "hybrid"), ("cem", "pretrained"), ("placebo2", "pretrained")]
    if args.models:
        models = args.models
    if args.pairs:
        pairs = [tuple(p.split(":")) for p in args.pairs]
    T = species_table()
    pre = np.load(os.path.join(FD, "pretrained_resnet34_imagenet_full.npz"))
    out = {"settings": {"CS": CS, "CF_GRID": CF_GRID, "CA_GRID": CA_GRID, "CS_A": CS_A, "CS_OFF": CS_OFF, "N_SPLITS": 1 if args.half else N_SPLITS, "MIN_SP": MIN_SP,
                        "N_BOOT": N_BOOT, "half": args.half, "SIGMAS": SIGMAS},
           "rows": {}, "tuned": {}, "n_attr": {}, "boot_store": {}, "phi": {}, "power": {}}
    store = {}  # (fold) -> {"y": Bte для S2 первого деления, "sp": виды фото, "series": {(m, k): оценки}}
    for fold in FOLDS:
        sel = R.selected_concepts(fold)
        held = np.setdiff1d(np.arange(112), sel)
        if SMOKE:
            held = held[:6]
        feats = {}
        for m in models:
            if m == "pretrained":
                feats[m] = {k: pre[k] for k in pre.files}
                continue
            p = os.path.join(FD, f"{m}_fold{fold}_full.npz")
            if os.path.exists(p):
                z = np.load(p)
                feats[m] = {k: z[k] for k in z.files}
        any_m = feats["pretrained"]
        Ytr, Yte = any_m["Y_train"].astype(int), any_m["Y_test"].astype(int)
        Ctr, Cte = any_m["C_train"].astype(int), any_m["C_test"].astype(int)
        for m, f in feats.items():  # фото в одном порядке у всех моделей
            assert np.array_equal(f["Y_train"].astype(int), Ytr) and np.array_equal(f["Y_test"].astype(int), Yte), m
        Atr, Ate = Ctr[:, sel].astype(np.float32), Cte[:, sel].astype(np.float32)
        Btr, Bte = Ctr[:, held], Cte[:, held]
        if args.half:
            sp = np.random.RandomState(100 + fold).permutation(200)
            splits = [(sp[:100], sp[100:])]
        else:
            rng = np.random.RandomState(500 + fold)
            splits = [tuple(np.split(rng.permutation(200), 2)) for _ in range(N_SPLITS)]

        def keep_mask(S1, S2):
            """Порог положительных в видах: ≥ MIN_SP положительных видов в каждой половине."""
            return (T[S1][:, held].sum(0) >= MIN_SP) & (T[S2][:, held].sum(0) >= MIN_SP)

        S1, S2 = splits[0]
        itr0 = np.isin(Ytr, S1)
        # подбор (C, s) на обучающей половине первого деления
        best = {}
        for m in feats:
            F = feats[m]["F_train"]
            best[(m, "F")] = tune(F[itr0], Btr[itr0], Ytr[itr0])[0]
            best[(m, "FA")], curve = tune_two(F[itr0], Atr[itr0], Btr[itr0], Ytr[itr0])
            out["tuned"][f"{m}|FA|{fold}"] = {"C_F": best[(m, "FA")][0], "C_A": best[(m, "FA")][1], "curve": curve}
            out["tuned"][f"{m}|F|{fold}"] = {"C": best[(m, "F")][0]}
            ite0u = np.isin(Yte, splits[0][0])
            best[(m, "FAu")], _ = tune_two(feats[m]["F_test"][ite0u], Ate[ite0u], Bte[ite0u], Yte[ite0u])
            out["tuned"][f"{m}|FAu|{fold}"] = {"C_F": best[(m, "FAu")][0], "C_A": best[(m, "FAu")][1]}
            if args.half:
                best[(m, "FtoA")] = tune(F[itr0], Atr[itr0].astype(int), Ytr[itr0])[0]
        best[("Aonly", "A")] = tune(Atr[itr0], Btr[itr0], Ytr[itr0], cs=CS_A)[0]
        ite0u = np.isin(Yte, splits[0][0])
        best[("Aonly", "Au")] = tune(Ate[ite0u], Bte[ite0u], Yte[ite0u], cs=CS_A)[0]
        print(f"fold {fold}: подобрано " + json.dumps({f"{a}|{b}": v for (a, b), v in best.items()}), flush=True)
        acc = {}
        for si, (S1, S2) in enumerate(splits):
            itr, ite = np.isin(Ytr, S1), np.isin(Yte, S2)
            keep = keep_mask(S1, S2)
            out["n_attr"][f"{fold}|{si}"] = int(keep.sum())
            Cs, _ = best[("Aonly", "A")]
            sc_a, ap_a = eval_attrs(Atr[itr], Btr[itr], Ate[ite], Bte[ite], Cs, keep=keep)
            acc.setdefault(("Aonly", "A"), []).append(ap_a)
            ser = {("Aonly", "A"): sc_a}
            trl, tel = stack_logits(Atr[itr], Btr[itr], Ytr[itr], Ate[ite], Cs)
            ite1 = np.isin(Yte, S1)  # test-фото обучающей половины видов: модель их не видела
            Cu, _ = best[("Aonly", "Au")]
            sc_au, ap_au = eval_attrs(Ate[ite1], Bte[ite1], Ate[ite], Bte[ite], Cu, keep=keep)
            acc.setdefault(("Aonly", "Au"), []).append(ap_au)
            ser[("Aonly", "Au")] = sc_au
            for m in feats:
                Ftr_all, Fte_all = feats[m]["F_train"], feats[m]["F_test"]
                C, _ = best[(m, "F")]
                sc, apv = eval_attrs(Ftr_all[itr], Btr[itr], Fte_all[ite], Bte[ite], C, keep=keep)
                acc.setdefault((m, "F"), []).append(apv)
                ser[(m, "F")] = sc
                cf, ca = best[(m, "FA")]
                sc, apv = eval_two(Ftr_all[itr], Atr[itr], Btr[itr], Fte_all[ite], Ate[ite], Bte[ite], cf, ca, keep)
                acc.setdefault((m, "FA"), []).append(apv)
                ser[(m, "FA")] = sc
                # STK: по атрибуту свой столбец логита B по A (×STK_SCALE после стандартизации)
                C, _ = best[(m, "F")]
                pt, pe = prep(Ftr_all[itr], Fte_all[ite])

                def stk(j):
                    lt, le = trl[:, j:j + 1], tel[:, j:j + 1]
                    mu, sd = lt.mean(), lt.std() + 1e-6
                    a = np.hstack([pt, STK_SCALE * (lt - mu) / sd])
                    b = np.hstack([pe, STK_SCALE * (le - mu) / sd])
                    return fit_scores(a, Btr[itr, j], b, C)
                sc = np.stack(Parallel(n_jobs=NJ)(delayed(stk)(j) for j in range(len(held))), 1)
                apv = np.array([ap_score(Bte[ite][:, j], sc[:, j]) if keep[j] and Bte[ite][:, j].sum() > 0
                                else np.nan for j in range(len(held))])
                acc.setdefault((m, "STK"), []).append(apv)
                # OFF: смещение — логит B по одним A (коэффициент 1), поверх обучается только F; C — свой (на первом делении)
                if (m, "OFF") not in best:
                    best[(m, "OFF")] = tune_off(pt, Btr[itr], trl, Ytr[itr])
                    out["tuned"][f"{m}|OFF|{fold}"] = {"C": best[(m, "OFF")]}
                Co = best[(m, "OFF")]
                sc = np.stack(Parallel(n_jobs=NJ)(delayed(fit_offset)(pt, Btr[itr, j].astype(float), trl[:, j], pe,
                                                                      tel[:, j], Co) for j in range(len(held))), 1)
                apv = np.array([ap_score(Bte[ite][:, j], sc[:, j]) if keep[j] and Bte[ite][:, j].sum() > 0
                                else np.nan for j in range(len(held))])
                acc.setdefault((m, "OFF"), []).append(apv)
                ser[(m, "OFF")] = sc
                # FAu: вложенная проба на test-фото обучающей половины (модель их не видела)
                cf, ca = best[(m, "FAu")]
                sc, apv = eval_two(Fte_all[ite1], Ate[ite1], Bte[ite1], Fte_all[ite], Ate[ite], Bte[ite], cf, ca, keep)
                acc.setdefault((m, "FAu"), []).append(apv)
                ser[(m, "FAu")] = sc
                if args.half:
                    C, _ = best[(m, "FtoA")]
                    _, apv = eval_attrs(Ftr_all[itr], Atr[itr].astype(int), Fte_all[ite], Ate[ite].astype(int), C)
                    acc.setdefault((m, "FtoA"), []).append(apv)
                    C, _ = best[(m, "F")]
                    # обычная проба на своих видах: S1 train → S1 test (ite1 — test-фото S1)
                    _, apv = eval_attrs(Ftr_all[itr], Btr[itr], Fte_all[ite1], Bte[ite1], C, keep=keep)
                    acc.setdefault((m, "Fseen"), []).append(apv)
                    # проба на фото, которых модель не видела: учится на test-фото S1, проверяется на test-фото S2
                    _, apv = eval_attrs(Fte_all[ite1], Bte[ite1], Fte_all[ite], Bte[ite], C, keep=keep)
                    acc.setdefault((m, "Funs"), []).append(apv)
            if si == 0:
                store[fold] = {"y": Bte[ite], "sp": Yte[ite], "keep": keep, "series": ser}
            print(f"fold {fold} деление {si}: готово ({int(keep.sum())} атрибутов по порогу в видах)", flush=True)
        for (m, k), lst in acc.items():
            arr = np.array(lst, float)
            out["rows"][f"{m}|{k}|{fold}"] = float(np.nanmean(np.nanmean(arr, 1)))
            print(f"  {m:14s} {k:6s} fold {fold}: AP {out['rows'][f'{m}|{k}|{fold}']:.4f}", flush=True)
        # N5: A6 исправлено (F, сравнение CEM − пустышка-2)
        a_m, b_m = ("cem_half", "placebo2_half") if args.half else ("cem", "placebo2")
        if (a_m, "F") in acc and (b_m, "F") in acc:
            diff = np.nanmean(np.array(acc[(a_m, "F")]), 0) - np.nanmean(np.array(acc[(b_m, "F")]), 0)
            maxphi = np.array([max(abs(np.corrcoef(T[:, j], T[:, s_])[0, 1]) for s_ in sel
                                   if T[:, s_].std() > 0) if T[:, j].std() > 0 else 0.0 for j in held])
            base = T[:, held].mean(0)
            ok = ~np.isnan(diff)
            Xr = np.column_stack([np.ones(ok.sum()), maxphi[ok], base[ok] - base[ok].mean()])
            coef = np.linalg.lstsq(Xr, diff[ok], rcond=None)[0]
            q25 = np.quantile(maxphi[ok], 0.25)
            out["phi"][str(fold)] = {"intercept_phi0_meanbase": float(coef[0]), "slope_phi": float(coef[1]),
                                     "slope_base": float(coef[2]), "q": np.quantile(maxphi[ok], [0, .1, .25, .5, .75, 1]).tolist(),
                                     "n_lt_0.1": int((maxphi[ok] < 0.1).sum()), "n": int(ok.sum()),
                                     "diff_lowq": float(np.mean(diff[ok][maxphi[ok] <= q25])),
                                     "diff_highq": float(np.mean(diff[ok][maxphi[ok] >= np.quantile(maxphi[ok], 0.75)]))}
        # N8: мощность (только --half, CEM без понятий)
        if args.half and "cemnoc_half" in feats:
            S1, S2 = splits[0]
            itr, ite = np.isin(Ytr, S1), np.isin(Yte, S2)
            keep = keep_mask(S1, S2)
            F = feats["cemnoc_half"]
            rngp = np.random.RandomState(7000 + fold)
            for sg in SIGMAS:
                inj_tr = Btr + sg * rngp.randn(*Btr.shape) * Btr.std(0, keepdims=True).clip(1e-3)
                inj_te = Bte + sg * rngp.randn(*Bte.shape) * Btr.std(0, keepdims=True).clip(1e-3)
                Ftr2, Fte2 = np.hstack([F["F_train"], inj_tr]), np.hstack([F["F_test"], inj_te])
                (cf, ca), _ = tune_two(Ftr2[itr], Atr[itr], Btr[itr], Ytr[itr])
                _, apv = eval_two(Ftr2[itr], Atr[itr], Btr[itr], Fte2[ite], Ate[ite], Bte[ite], cf, ca, keep)
                out["power"][f"{sg:g}|{fold}"] = float(np.nanmean(apv))
                print(f"  мощность σ = {sg:g} fold {fold}: AP[F′, A] {np.nanmean(apv):.4f}", flush=True)
    # сводка
    rows = out["rows"]
    aonly = {f: rows[f"Aonly|A|{f}"] for f in FOLDS if f"Aonly|A|{f}" in rows}
    print(f"\nодни A_true: AP {np.mean(list(aonly.values())):.4f}")
    out["valid"] = {}
    for m in models:
        fs = [f for f in FOLDS if f"{m}|FA|{f}" in rows]
        if not fs:
            continue
        aonly_u = {f: rows[f"Aonly|Au|{f}"] for f in FOLDS if f"Aonly|Au|{f}" in rows}
        for k in ("FA", "STK", "OFF", "FAu"):
            base = aonly_u if k == "FAu" else aonly
            d = [rows[f"{m}|{k}|{f}"] - base[f] for f in fs]
            out["valid"][f"{m}|{k}"] = {"delta": interval(d), "valid": bool(np.mean(d) >= -0.002)}
            print(f"  Δ({k} | A) {m:14s}: {interval(d)[0]:+.4f} [{interval(d)[1]:+.4f}; {interval(d)[2]:+.4f}] "
                  f"{'годна' if np.mean(d) >= -0.002 else 'НЕГОДНА'}", flush=True)
    out["paired"] = {}
    print("\nпарные разницы по повторам, AP [95% t]:")
    for k in ("F", "FA", "STK", "OFF", "FAu", "FtoA", "Fseen", "Funs"):
        for a, b in pairs:
            fs = [f for f in FOLDS if f"{a}|{k}|{f}" in rows and f"{b}|{k}|{f}" in rows]
            if len(fs) < 2:
                continue
            d = [rows[f"{a}|{k}|{f}"] - rows[f"{b}|{k}|{f}"] for f in fs]
            out["paired"][f"{a}-{b}|{k}"] = {"n": len(fs), "ap": interval(d), "per_fold": d}
            iv = interval(d)
            print(f"  {k:6s} {a:14s} − {b:14s} n={len(fs)}: {iv[0]:+.4f} [{iv[1]:+.4f}; {iv[2]:+.4f}]  "
                  f"по повторам {np.round(d, 4).tolist()}", flush=True)
    # N4: двухуровневый бутстрэп (повторы × проверочные виды первого деления)
    rngb = np.random.RandomState(12345)
    fl = sorted(store)
    keys = [(a, b, k) for (a, b) in pairs for k in ("F", "FA", "OFF", "FAu")
            if all((x, k) in store[fl[0]]["series"] for x in (a, b))]
    rj = np.random.RandomState(777)
    orders = {f: {mk: np.argsort(-(sc + 1e-9 * (sc.std(0, keepdims=True) + 1e-12) * rj.randn(*sc.shape)), axis=0)
                  for mk, sc in store[f]["series"].items()} for f in fl}

    def one_draw(seed):
        r = np.random.RandomState(seed)
        pick = r.choice(fl, len(fl))
        vals = {key: [] for key in keys}
        for f in pick:
            st = store[f]
            uniq = np.unique(st["sp"])
            w = np.zeros(len(st["sp"]))
            for s_ in r.choice(uniq, len(uniq)):
                w[st["sp"] == s_] += 1
            cache = {}
            for (a, b, k) in keys:
                for x in (a, b):
                    if (x, k) not in cache:
                        o = orders[f][(x, k)]
                        cache[(x, k)] = np.nanmean([weighted_ap(o[:, j], st["y"][:, j], w) for j in range(st["y"].shape[1])
                                                    if st["keep"][j]])
                vals[(a, b, k)].append(cache[(a, k)] - cache[(b, k)])
        return {key: float(np.mean(v)) for key, v in vals.items()}
    draws = Parallel(n_jobs=NJ)(delayed(one_draw)(int(sd)) for sd in rngb.randint(0, 2**31 - 1, N_BOOT))
    print("\nдвухуровневый бутстрэп (повторы × виды, первое деление), AP [2.5%; 97.5%]:")
    for key in keys:
        v = np.array([d_[key] for d_ in draws])
        lo, hi = np.percentile(v, [2.5, 97.5])
        out["boot_store"][f"{key[0]}-{key[1]}|{key[2]}"] = [float(v.mean()), float(lo), float(hi)]
        print(f"  {key[2]:3s} {key[0]:14s} − {key[1]:14s}: {v.mean():+.4f} [{lo:+.4f}; {hi:+.4f}]", flush=True)
    if out["phi"]:
        ph = out["phi"].values()
        print("\nA6 (исправлено): свободный член при max|φ| = 0 и средней доле «да» "
              f"{interval([p['intercept_phi0_meanbase'] for p in ph])}; наклон по max|φ| {interval([p['slope_phi'] for p in ph])}; "
              f"нижняя четверть max|φ|: {interval([p['diff_lowq'] for p in ph])}; верхняя: {interval([p['diff_highq'] for p in ph])}; "
              f"B с max|φ| < 0.1: {[p['n_lt_0.1'] for p in ph]} из {[p['n'] for p in ph]}; квантили max|φ| (повтор 0) "
              f"{np.round(list(out['phi'].values())[0]['q'], 3).tolist()}", flush=True)
    if out["power"]:
        for sg in SIGMAS:
            d = [out["power"][f"{sg:g}|{f}"] - rows[f"cemnoc_half|FA|{f}"] for f in FOLDS if f"{sg:g}|{f}" in out["power"]]
            out["power"][f"delta_{sg:g}"] = interval(d)
            print(f"мощность: σ = {sg:g}: AP[F′, A] − AP[F, A] у CEM без понятий {interval(d)}", flush=True)
    print(f"атрибутов по порогу в видах: {sorted(set(out['n_attr'].values()))}")
    name = ("smoke_" if SMOKE else "") + ("analysis_nested_half" if args.half else "analysis_nested") + args.tag + ".json"
    json.dump(out, open(os.path.join(R.OUT, name), "w"), ensure_ascii=False, indent=1)
    print("NESTED_DONE", flush=True)


if __name__ == "__main__":
    main()
