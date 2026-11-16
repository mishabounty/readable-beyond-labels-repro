#!/usr/bin/env python
"""Проба на непройденных видах по протоколу (план experiments/PLAN_2026-10-30_checks.md, пункты A1–A4, A6, C).

Учимся на фото train одной половины видов, проверяем на фото test другой: атрибут нужно прочитать по внешности
незнакомой пробе птицы. Исправляет замечания рецензий 30.10:
  A1 — канал A → B: входы F (узкое место), A_true (28 настоящих обучаемых атрибутов, уровень вида), [F, A_true],
       Â (28 вероятностей обучаемых атрибутов, предсказанных по F на обучающей половине), avgpool, [avgpool, A_true];
       у исходной ResNet — F и [F, A_true]; отдельно одни A_true;
  A2 — штраф C из {0.001, 0.01, 0.1, 1, 10} подбирается по средней AP в GroupKFold (3 части по видам) внутри
       обучающей половины первого деления, один на модель × повтор × вход;
  A3 — 10 случайных делений видов на повтор (в режиме --half одно: модель обучена только на первой половине);
  A4 — кластерный бутстрэп по проверочным видам (точность; 200 выборок; одна выборка видов для всех моделей);
  A6 — разница CEM − пустышка-2 по AP атрибута против max|φ| с обучаемыми и доли «да»: наклон и свободный член
       (величина при φ = 0), по повторам.
Режим --half (пункт C): модели cem_half / placebo2_half / cemnoc_half обучены только на видах S1 =
RandomState(100 + fold).permutation(200)[:100]; проба учится на S1 и проверяется на S2 — виды, не виденные ни моделью,
ни пробой.
Итог: results/cub_subsample_0.25/analysis_unseen[_half].json и сводка в stdout. Только CPU.
Запуск: NJ=12 python analyze_unseen.py [--half]
"""

import argparse
import json
import os
import pickle
import sys

import numpy as np
from joblib import Parallel, delayed
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupKFold

sys.path.insert(0, os.path.expanduser("~/cem_repro"))
import run_subsample_2022 as R  # noqa: E402

FD = os.path.join(R.OUT, "features")
FOLDS = [0, 1, 2, 3, 4]
NJ = int(os.environ.get("NJ", "8"))
CS = [0.001, 0.01, 0.1, 1.0, 10.0]
N_SPLITS = 10
N_BOOT = 200
MIN_POS = 5
# Чувствительность (добавлено 30.10 после повтора 0 планового прогона: C почти везде упёрся в нижний край сетки 0.001,
# а [F, A_true] при общем штрафе давал AP ниже, чем одни A_true, — проба недоиспользует A):
#   CGRID=wide — сетка C от 1e-5 до 10 (по плану — от 0.001);
#   ASCALE=s   — после стандартизации столбцы A_true во входах [F, A_true] и [avgpool, A_true] умножаются на s, то есть
#                их штраф в s² раз слабее штрафа F (по плану s = 1, общий штраф).
CGRID = os.environ.get("CGRID", "plan")
ASCALE = float(os.environ.get("ASCALE", "1"))
if CGRID == "wide":
    CS = [1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0]
N_A = 28  # число обучаемых атрибутов (столбцы A_true стоят последними во входах FA и AA)
SMOKE = os.environ.get("SMOKE") == "1"  # пробный запуск: 1 повтор, 1 деление, 1 C, 6 атрибутов
if SMOKE:
    FOLDS, N_SPLITS, CS, N_BOOT = [0, 1], 1, [0.1], 5


def standardize(tr, *others, n_a=0):
    """Стандартизация по обучающей части; при n_a > 0 последние n_a столбцов (A_true) ещё умножаются на ASCALE."""
    mu, sd = tr.mean(0), tr.std(0) + 1e-6
    out = [(x - mu) / sd for x in (tr,) + others]
    if n_a and ASCALE != 1.0:
        for x in out:
            x[:, -n_a:] *= ASCALE
    return out


def fit_one(Xtr, ytr, Xte, yte, C):
    """Точность, AP (или None) и вектор верных ответов на проверочных фото."""
    if ytr.min() == ytr.max():
        pred = np.full_like(yte, ytr[0])
        return float((pred == yte).mean()), None, (pred == yte).astype(np.int8)
    m = LogisticRegression(C=C, max_iter=1000).fit(Xtr, ytr)
    pred = m.predict(Xte)
    ap = float(average_precision_score(yte, m.decision_function(Xte))) if yte.sum() >= MIN_POS else None
    return float((pred == yte).mean()), ap, (pred == yte).astype(np.int8)


def tune_C(X, Yb, groups, n_a=0):
    """Один C по средней AP в GroupKFold по видам (3 части) внутри обучающей половины; возвращает C и кривую AP(C)."""
    parts = []
    for tr, va in GroupKFold(3).split(X, groups=groups):
        Xtr, Xva = standardize(X[tr], X[va], n_a=n_a)
        parts.append((Xtr, Xva, tr, va))

    def score(C, k, j):
        Xtr, Xva, tr, va = parts[k]
        ytr, yva = Yb[tr, j], Yb[va, j]
        if ytr.min() == ytr.max() or yva.sum() < MIN_POS:
            return None
        m = LogisticRegression(C=C, max_iter=1000).fit(Xtr, ytr)
        return float(average_precision_score(yva, m.decision_function(Xva)))

    tasks = [(C, k, j) for C in CS for k in range(len(parts)) for j in range(Yb.shape[1])]
    res = Parallel(n_jobs=NJ)(delayed(score)(*t) for t in tasks)
    by = {C: [] for C in CS}
    for (C, _, _), r in zip(tasks, res):
        if r is not None:
            by[C].append(r)
    curve = {str(c): (float(np.mean(by[c])) if by[c] else None) for c in CS}
    return max(CS, key=lambda c: (np.mean(by[c]) if by[c] else -1, -abs(np.log10(c)))), curve


def ahat_col(Ftr, ya, groups, Fte):
    """Вероятность одного обучаемого атрибута по F: на обучающих фото — вне выборки (GroupKFold по видам, 3 части),
    чтобы Â на обучении не было точнее, чем на проверке; на проверочных — регрессия по всей обучающей половине."""
    if ya.min() == ya.max():
        return np.full(len(ya), float(ya[0])), np.full(len(Fte), float(ya[0]))
    tr_col = np.empty(len(ya))
    for a, b in GroupKFold(3).split(Ftr, groups=groups):
        if ya[a].min() == ya[a].max():
            tr_col[b] = float(ya[a][0])
        else:
            tr_col[b] = LogisticRegression(C=1.0, max_iter=1000).fit(Ftr[a], ya[a]).predict_proba(Ftr[b])[:, 1]
    te_col = LogisticRegression(C=1.0, max_iter=1000).fit(Ftr, ya).predict_proba(Fte)[:, 1]
    return tr_col, te_col


def load_feats(name):
    z = np.load(os.path.join(FD, name))
    out = {k: z[k] for k in z.files}
    return out


def species_table():
    L = {}
    for d in pickle.load(open(os.path.join(R.BASE_DIR, "train.pkl"), "rb")):
        L[d["class_label"]] = np.array(d["attribute_label"])
    return np.stack([L[k] for k in range(200)]).astype(float)  # 200 × 112, уровень вида


def interval(d):
    d = np.asarray(d, float)
    if len(d) < 2:
        return [float(np.mean(d)) if len(d) else float("nan"), float("nan"), float("nan")]
    h = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    return [float(d.mean()), float(d.mean() - h), float(d.mean() + h)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--half", action="store_true")
    args = ap.parse_args()
    if args.half:
        models = ["cem_half", "placebo2_half", "cemnoc_half"]
        pairs = [("cem_half", "placebo2_half"), ("cem_half", "cemnoc_half"), ("placebo2_half", "cemnoc_half")]
    else:
        models = ["cem", "placebo", "placebo2", "cemnoc", "hybrid", "noconcept"]
        pairs = [("cem", "placebo2"), ("cem", "placebo"), ("placebo", "placebo2"), ("cem", "cemnoc"),
                 ("placebo2", "cemnoc"), ("cemnoc", "noconcept"), ("cem", "hybrid"), ("hybrid", "noconcept"),
                 ("cem", "pretrained"), ("placebo2", "pretrained")]
    T = species_table()
    out = {"rows": {}, "per_attr": {}, "C": {}, "C_curve": {}, "boot": {}, "split_sd": {},
           "settings": {"CS": CS, "ASCALE": ASCALE, "N_SPLITS": N_SPLITS, "N_BOOT": N_BOOT, "half": args.half}}
    pre = load_feats("pretrained_resnet34_imagenet_full.npz") if not args.half else None
    for fold in FOLDS:
        sel = R.selected_concepts(fold)
        held = np.setdiff1d(np.arange(112), sel)
        if SMOKE:
            held = held[:6]
        feats = {}
        for m in models:
            p = os.path.join(FD, f"{m}_fold{fold}_full.npz")
            if os.path.exists(p):
                feats[m] = load_feats(f"{m}_fold{fold}_full.npz")
        if pre is not None:
            feats["pretrained"] = {"F_train": pre["F_train"], "F_test": pre["F_test"], "C_train": pre["C_train"],
                                   "C_test": pre["C_test"], "Y_train": pre["Y_train"], "Y_test": pre["Y_test"]}
        if not feats:
            continue
        any_m = next(iter(feats.values()))
        Ytr, Yte = any_m["Y_train"].astype(int), any_m["Y_test"].astype(int)
        Ctr, Cte = any_m["C_train"].astype(int), any_m["C_test"].astype(int)
        Atr, Ate = Ctr[:, sel].astype(np.float32), Cte[:, sel].astype(np.float32)  # A_true на фото
        Btr, Bte = Ctr[:, held], Cte[:, held]
        # деления видов: в режиме --half одно (S1 — виды обучения модели), иначе 10 случайных
        splits = []
        if args.half:
            sp = np.random.RandomState(100 + fold).permutation(200)
            splits.append((set(sp[:100].tolist()), set(sp[100:].tolist())))
        else:
            rng = np.random.RandomState(500 + fold)
            for _ in range(N_SPLITS):
                sp = rng.permutation(200)
                splits.append((set(sp[:100].tolist()), set(sp[100:].tolist())))

        def featsets(m):
            f = feats[m]
            fs = {"F": (f["F_train"], f["F_test"]),
                  "FA": (np.hstack([f["F_train"], Atr]), np.hstack([f["F_test"], Ate]))}
            if "A_train" in f:
                fs["A"] = (f["A_train"], f["A_test"])
                fs["AA"] = (np.hstack([f["A_train"], Atr]), np.hstack([f["A_test"], Ate]))
            return fs

        # выбор C на первом делении (обучающая половина), один на модель × вход
        s_tr0 = np.isin(Ytr, list(splits[0][0]))
        jobs = [(m, k) for m in feats for k in featsets(m)] + [("Aonly", "A_true")]
        Cbest = {}
        for m, k in jobs:
            Xtr = Atr if m == "Aonly" else featsets(m)[k][0]
            Cbest[(m, k)], out["C_curve"][f"{m}|{k}|{fold}"] = tune_C(Xtr[s_tr0], Btr[s_tr0], Ytr[s_tr0],
                                                                      n_a=N_A if k in ("FA", "AA") else 0)
            out["C"][f"{m}|{k}|{fold}"] = Cbest[(m, k)]
        if "cem" in feats or "cem_half" in feats:
            for m in [x for x in ("cem", "cem_half") if x in feats]:
                Cbest[(m, "Ahat")] = Cbest[(m, "F")]
        print(f"fold {fold}: C выбраны {json.dumps({f'{a}|{b}': c for (a, b), c in Cbest.items()})}", flush=True)
        acc_split = {}
        ap_attr = {}
        for si, (Str, Ste) in enumerate(splits):
            itr, ite = np.isin(Ytr, list(Str)), np.isin(Yte, list(Ste))
            correct = {}
            for (m, k), C in Cbest.items():
                if m == "Aonly":
                    Xtr, Xte = Atr, Ate
                elif k == "Ahat":
                    # Â: 28 вероятностей обучаемых атрибутов по F (обучающая половина; на ней — вне выборки)
                    Ftr, Fte = standardize(feats[m]["F_train"][itr], feats[m]["F_test"][ite])
                    cols = Parallel(n_jobs=NJ)(delayed(ahat_col)(Ftr, Atr[itr, a].astype(int), Ytr[itr], Fte)
                                               for a in range(Atr.shape[1]))
                    Xtr, Xte = np.stack([c[0] for c in cols], 1), np.stack([c[1] for c in cols], 1)
                    Xtr, Xte = standardize(Xtr, Xte)
                    res = Parallel(n_jobs=NJ)(delayed(fit_one)(Xtr, Btr[itr, j], Xte, Bte[ite, j], C)
                                              for j in range(len(held)))
                    acc_split.setdefault((m, k), []).append(np.mean([r[0] for r in res]))
                    ap_attr.setdefault((m, k), []).append([r[1] for r in res])
                    correct[(m, k)] = np.stack([r[2] for r in res], 1)
                    continue
                else:
                    Xtr, Xte = featsets(m)[k][0][itr], featsets(m)[k][1][ite]
                if m == "Aonly":
                    Xtr, Xte = Xtr[itr], Xte[ite]
                Xtr, Xte = standardize(Xtr, Xte, n_a=N_A if k in ("FA", "AA") else 0)
                res = Parallel(n_jobs=NJ)(delayed(fit_one)(Xtr, Btr[itr, j], Xte, Bte[ite, j], C)
                                          for j in range(len(held)))
                acc_split.setdefault((m, k), []).append(np.mean([r[0] for r in res]))
                ap_attr.setdefault((m, k), []).append([r[1] for r in res])
                correct[(m, k)] = np.stack([r[2] for r in res], 1)
            # кластерный бутстрэп по проверочным видам (точность), одна выборка видов для всех моделей
            sp_te = Yte[ite]
            uniq = np.unique(sp_te)
            idx_by = {s: np.where(sp_te == s)[0] for s in uniq}
            rngb = np.random.RandomState(1000 * fold + si)
            draws = [np.concatenate([idx_by[s] for s in rngb.choice(uniq, len(uniq))]) for _ in range(N_BOOT)]
            for a, b in pairs:
                for k in ("F", "FA"):
                    if (a, k) not in correct or (b, k) not in correct:
                        continue
                    d = correct[(a, k)].astype(float).mean(1) - correct[(b, k)].astype(float).mean(1)
                    bs = [100 * d[ix].mean() for ix in draws]
                    out["boot"].setdefault(f"{a}-{b}|{k}|{fold}", []).append(
                        [float(100 * d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))])
            print(f"fold {fold} деление {si}: готово", flush=True)
        for (m, k), accs in acc_split.items():
            aps = np.array([[np.nan if v is None else v for v in row] for row in ap_attr[(m, k)]], float)
            ap_mean_attr = np.nanmean(aps, 0)  # по делениям
            ap_split = np.nanmean(aps, 1)  # по атрибутам
            out["rows"][f"{m}|{k}|{fold}"] = [float(np.mean(accs)), float(np.nanmean(ap_split))]
            out["split_sd"][f"{m}|{k}|{fold}"] = [float(np.std(accs)), float(np.nanstd(ap_split))]
            out["per_attr"][f"{m}|{k}|{fold}"] = [None if np.isnan(v) else float(v) for v in ap_mean_attr]
            print(f"  {m:14s} {k:6s} fold {fold}: точность {100 * np.mean(accs):.2f}, AP {np.nanmean(ap_split):.4f}, "
                  f"C {Cbest[(m, k)]}", flush=True)
        # A6: разница по атрибуту против max|φ| и доли «да»
        a_m, b_m = ("cem_half", "placebo2_half") if args.half else ("cem", "placebo2")
        ka, kb = f"{a_m}|F|{fold}", f"{b_m}|F|{fold}"
        if ka in out["per_attr"] and kb in out["per_attr"]:
            diff = np.array([np.nan if (x is None or y is None) else x - y
                             for x, y in zip(out["per_attr"][ka], out["per_attr"][kb])])
            maxphi = np.array([max(abs(np.corrcoef(T[:, j], T[:, s])[0, 1]) for s in sel
                                   if T[:, s].std() > 0 and T[:, j].std() > 0) if T[:, j].std() > 0 else 0.0
                               for j in held])
            base = Bte.mean(0)
            ok = ~np.isnan(diff)
            Xr = np.column_stack([np.ones(ok.sum()), maxphi[ok], base[ok]])
            coef = np.linalg.lstsq(Xr, diff[ok], rcond=None)[0]
            out.setdefault("phi", {})[str(fold)] = {"intercept": float(coef[0]), "slope_phi": float(coef[1]),
                                                    "slope_base": float(coef[2]),
                                                    "spearman": float(stats.spearmanr(maxphi[ok], diff[ok])[0])}
    # сводка: парные разницы по повторам
    out["paired"] = {}
    print("\nпарные разницы по повторам (среднее по делениям внутри повтора): точность (п.) [95%] | AP [95%]")
    for k in ("F", "FA", "Ahat", "A", "AA"):
        for a, b in pairs:
            fs = [f for f in FOLDS if f"{a}|{k}|{f}" in out["rows"] and f"{b}|{k}|{f}" in out["rows"]]
            if len(fs) < 2:
                continue
            da = [100 * (out["rows"][f"{a}|{k}|{f}"][0] - out["rows"][f"{b}|{k}|{f}"][0]) for f in fs]
            dp = [out["rows"][f"{a}|{k}|{f}"][1] - out["rows"][f"{b}|{k}|{f}"][1] for f in fs]
            ia, ip = interval(da), interval(dp)
            out["paired"][f"{a}-{b}|{k}"] = {"n": len(fs), "acc": ia, "ap": ip}
            print(f"  {k:5s} {a:14s} − {b:14s} n={len(fs)}: {ia[0]:+.2f} [{ia[1]:+.2f}; {ia[2]:+.2f}] | "
                  f"{ip[0]:+.4f} [{ip[1]:+.4f}; {ip[2]:+.4f}]", flush=True)
    # правила A1 / C: разница CEM − пустышка-2 по AP на F и на [F, A_true]; доля, которая остаётся при известных A
    a_m, b_m = ("cem_half", "placebo2_half") if args.half else ("cem", "placebo2")
    gF, gFA = out["paired"].get(f"{a_m}-{b_m}|F"), out["paired"].get(f"{a_m}-{b_m}|FA")
    if gF and gFA:
        fs = [f for f in FOLDS if all(f"{x}|{k}|{f}" in out["rows"] for x in (a_m, b_m) for k in ("F", "FA"))]
        ratio = [(out["rows"][f"{a_m}|FA|{f}"][1] - out["rows"][f"{b_m}|FA|{f}"][1]) /
                 (out["rows"][f"{a_m}|F|{f}"][1] - out["rows"][f"{b_m}|F|{f}"][1]) for f in fs]
        out["rule_A1"] = {"gap_F_ap": gF["ap"], "gap_FA_ap": gFA["ap"], "ratio_FA_to_F_per_fold": ratio,
                          "verdict": ("канал A → B (интервал на [F, A_true] задевает 0)" if gFA["ap"][1] <= 0 <= gFA["ap"][2]
                                      else ("сверх A" if gFA["ap"][1] > 0 else "пустышка-2 выше при известных A"))}
        print(f"\nПРАВИЛО A1/C: {a_m} − {b_m} по AP: на F {gF['ap'][0]:+.4f} [{gF['ap'][1]:+.4f}; {gF['ap'][2]:+.4f}], "
              f"на [F, A_true] {gFA['ap'][0]:+.4f} [{gFA['ap'][1]:+.4f}; {gFA['ap'][2]:+.4f}]; доля, что остаётся "
              f"при известных A, по повторам {np.round(ratio, 2).tolist()} → {out['rule_A1']['verdict']}", flush=True)
    for m in ("cem", "cem_half"):
        ks = [k for k in ("F", "Ahat", "FA") if all(f"{m}|{k}|{f}" in out["rows"] for f in FOLDS)]
        if ks:
            print(f"{m}: AP на непройденных видах по входам " + ", ".join(
                f"{k} {np.mean([out['rows'][f'{m}|{k}|{f}'][1] for f in FOLDS]):.4f}" for k in ks), flush=True)
    aonly = [out["rows"][f"Aonly|A_true|{f}"] for f in FOLDS if f"Aonly|A_true|{f}" in out["rows"]]
    if aonly:
        print(f"\nодни A_true: точность {100 * np.mean([r[0] for r in aonly]):.2f}, AP {np.mean([r[1] for r in aonly]):.4f}")
    if "phi" in out:
        ints = [v["intercept"] for v in out["phi"].values()]
        sl = [v["slope_phi"] for v in out["phi"].values()]
        print(f"A6: свободный член (разница AP при max|φ| = 0) {interval(ints)}; наклон по max|φ| {interval(sl)}")
    bw = {}
    for key, rows in out["boot"].items():
        pair_k = key.rsplit("|", 1)[0]
        bw.setdefault(pair_k, []).extend([r[2] - r[1] for r in rows])
    print("средняя ширина 95% бутстрэп-интервала по видам (точность, п.):",
          json.dumps({k: round(float(np.mean(v)), 2) for k, v in bw.items()}, ensure_ascii=False))
    tag = ("_wideC" if CGRID == "wide" else "") + (f"_as{ASCALE:g}" if ASCALE != 1.0 else "")
    name = ("smoke_" if SMOKE else "") + ("analysis_unseen_half" if args.half else "analysis_unseen") + tag + ".json"
    edge = [k for k, c in out["C"].items() if c in (min(CS), max(CS))]
    print(f"C на краю сетки {CS[0]:g}…{CS[-1]:g}: {len(edge)} из {len(out['C'])}", flush=True)
    json.dump(out, open(os.path.join(R.OUT, name), "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
