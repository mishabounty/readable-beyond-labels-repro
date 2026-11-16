#!/usr/bin/env python
"""Дополнительные проверки опыта CEM (29.10.2026) на сохранённых признаках (extract_features.py), только CPU.

Для каждой модели и разбиения, на двух местах — узкое место (F) и выход ResNet перед ним (A, avgpool, одно место у
всех моделей) — три способа чтения 84 непройденных атрибутов логистической регрессией на стандартизованных
признаках:
  std    — обучение на train, проверка на test, C ∈ {0.01, 0.1, 1} по val (как проба 2);
  unseen — проба на непройденных видах: виды делятся пополам (зерно 100 + fold); учимся на фото train одной половины
           видов, проверяем на фото test другой половины, и наоборот; C = 0.1. Атрибут нужно читать по внешности,
           а не узнавать вид: метки атрибутов заданы на уровне вида, и узнавание вида тут не помогает;
  halves — перекрёстная проба на половинах test, разделённых внутри каждого вида (зерно fold); C = 0.1. Нет обучающих
           фото, которые сеть запомнила.
Меры: средняя точность и AP (атрибуты с ≥ 5 положительными в проверочной части). Ориентир — исходная ResNet-34 с
нормировкой ImageNet (std). Итог — парные разницы по повторам (среднее, 95% t-интервал), описательно (план §14.1:
вердикты только по главной конечной точке). Плюс разбор по атрибутам: связь разницы CEM − пустышка с max|φ| между
атрибутом и 28 обучаемыми (на уровне вида), двухступенчато: корреляция Спирмена в каждом повторе, затем t по повторам.
Запуск: python analyze_extra.py → results/analysis_extra.json и сводка в stdout.
"""

import json
import os
import pickle
import sys

import numpy as np
from joblib import Parallel, delayed
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

sys.path.insert(0, os.path.expanduser("~/cem_repro"))
import run_subsample_2022 as R  # noqa: E402

FD = os.path.join(R.OUT, "features")
MODELS = ["cem", "placebo", "placebo2", "cemnoc", "hybrid", "noconcept"]
FOLDS = [0, 1, 2, 3, 4]
NJ = int(os.environ.get("NJ", "4"))


def std_feats(tr, *others):
    mu, sd = tr.mean(0), tr.std(0) + 1e-6
    return [(x - mu) / sd for x in (tr,) + others]


def fit_eval(Xtr, ytr, Xte, yte, C):
    if ytr.min() == ytr.max():
        acc = float((yte == ytr[0]).mean())
        return acc, None
    m = LogisticRegression(C=C, max_iter=3000).fit(Xtr, ytr)
    acc = float((m.predict(Xte) == yte).mean())
    ap = float(average_precision_score(yte, m.decision_function(Xte))) if yte.sum() >= 5 else None
    return acc, ap


def summarize(rows):
    accs = [r[0] for r in rows]
    aps = [r[1] for r in rows if r[1] is not None]
    return float(np.mean(accs)), float(np.mean(aps)) if aps else float("nan")


def probe_std(F, C, held):
    tr, va, te = std_feats(F["train"], F["val"], F["test"])

    def one(j):
        ytr, yva, yte = C["train"][:, j], C["val"][:, j], C["test"][:, j]
        best = None
        for c in (0.01, 0.1, 1.0):
            if ytr.min() == ytr.max():
                break
            m = LogisticRegression(C=c, max_iter=3000).fit(tr, ytr)
            v = float((m.predict(va) == yva).mean())
            if best is None or v > best[0]:
                best = (v, c)
        return fit_eval(tr, ytr, te, yte, best[1] if best else 1.0)

    return summarize(Parallel(n_jobs=NJ)(delayed(one)(j) for j in held))


def probe_unseen(F, C, Y, held, fold):
    rng = np.random.RandomState(100 + fold)
    sp = rng.permutation(200)
    halves = (set(sp[:100].tolist()), set(sp[100:].tolist()))
    out = []
    for a, b in ((0, 1), (1, 0)):
        itr = np.isin(Y["train"], list(halves[a]))
        ite = np.isin(Y["test"], list(halves[b]))
        tr, te = std_feats(F["train"][itr], F["test"][ite])
        rows = Parallel(n_jobs=NJ)(delayed(fit_eval)(tr, C["train"][itr, j], te, C["test"][ite, j], 0.1)
                                   for j in held)
        out.append(summarize(rows))
    return float(np.mean([o[0] for o in out])), float(np.nanmean([o[1] for o in out]))


def probe_halves(F, C, Y, held, fold):
    rng = np.random.RandomState(fold)
    h = np.zeros(len(Y["test"]), dtype=bool)
    for k in np.unique(Y["test"]):
        idx = np.where(Y["test"] == k)[0]
        h[rng.permutation(idx)[: len(idx) // 2]] = True
    out = []
    for a in (h, ~h):
        tr, te = std_feats(F["test"][a], F["test"][~a])
        rows = Parallel(n_jobs=NJ)(delayed(fit_eval)(tr, C["test"][a, j], te, C["test"][~a, j], 0.1) for j in held)
        out.append(summarize(rows))
    return float(np.mean([o[0] for o in out])), float(np.nanmean([o[1] for o in out]))


def load(path, key):
    z = np.load(path)
    return ({s: z[f"{key}_{s}"].astype(np.float32) for s in ("train", "val", "test")},
            {s: z[f"C_{s}"].astype(np.int64) for s in ("train", "val", "test")},
            {s: z[f"Y_{s}"].astype(np.int64) for s in ("train", "val", "test")})


def interval(d):
    d = np.asarray(d, float)
    if len(d) < 2:
        return float(d.mean()) if len(d) else float("nan"), float("nan"), float("nan")
    h = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    return float(d.mean()), float(d.mean() - h), float(d.mean() + h)


def phi_analysis():
    """Двухступенчато: в каждом повторе корреляция Спирмена между max|φ|(B_j, A) и разницей по атрибуту."""
    L = {}
    for d in pickle.load(open(os.path.join(R.BASE_DIR, "train.pkl"), "rb")):
        L[d["class_label"]] = np.array(d["attribute_label"])
    T = np.stack([L[k] for k in range(200)]).astype(float)  # 200 × 112, уровень вида
    per = {}
    for line in open(os.path.join(R.OUT, "probe_per_concept.jsonl")):
        r = json.loads(line)
        per[(r["model"], r["fold"])] = {c["j"]: c["acc_tune"] for c in r["concepts"]}
    res = {}
    for b in ("placebo", "placebo2"):
        rhos = []
        for f in FOLDS:
            if ("cem", f) not in per or (b, f) not in per:
                continue
            sel = R.selected_concepts(f)
            held = [j for j in per[("cem", f)] if j in per[(b, f)]]
            maxphi = []
            for j in held:
                cs = [abs(np.corrcoef(T[:, j], T[:, a])[0, 1]) for a in sel if T[:, a].std() > 0 and T[:, j].std() > 0]
                maxphi.append(max(cs) if cs else 0.0)
            diff = [per[("cem", f)][j] - per[(b, f)][j] for j in held]
            rhos.append(stats.spearmanr(maxphi, diff)[0])
        m, lo, hi = interval(rhos)
        res[b] = {"rho_per_fold": [float(x) for x in rhos], "mean": m, "lo": lo, "hi": hi}
    return res


def main():
    out = {"rows": {}}
    pre = os.path.join(FD, "pretrained_resnet34_imagenet_full.npz")
    if os.path.exists(pre):
        F, C, Y = load(pre, "F")
        for f in FOLDS:
            held = np.setdiff1d(np.arange(112), R.selected_concepts(f))
            out["rows"][f"pretrained_imagenet|F|std|{f}"] = probe_std(F, C, held)
            print("pretrained_imagenet", f, out["rows"][f"pretrained_imagenet|F|std|{f}"], flush=True)
    for m in MODELS:
        for f in FOLDS:
            p = os.path.join(FD, f"{m}_fold{f}_full.npz")
            if not os.path.exists(p):
                print("нет", p, flush=True)
                continue
            held = np.setdiff1d(np.arange(112), R.selected_concepts(f))
            cache = os.path.join(R.OUT, "extra_cache", f"{m}_{f}.json")  # готовые модели не пересчитываются
            if os.path.exists(cache):
                out["rows"].update(json.load(open(cache)))
                continue
            got = {}
            for key in ("F", "A"):
                F, C, Y = load(p, key)
                r = {"std": probe_std(F, C, held), "unseen": probe_unseen(F, C, Y, held, f),
                     "halves": probe_halves(F, C, Y, held, f)}
                for kind, v in r.items():
                    got[f"{m}|{key}|{kind}|{f}"] = v
                print(m, key, f, {k: (round(100 * v[0], 2), round(v[1], 3)) for k, v in r.items()}, flush=True)
            os.makedirs(os.path.dirname(cache), exist_ok=True)
            json.dump(got, open(cache, "w"))
            out["rows"].update(got)
    pairs = [("cem", "placebo"), ("cem", "placebo2"), ("placebo", "placebo2"), ("cem", "cemnoc"),
             ("placebo2", "cemnoc"), ("cemnoc", "noconcept"), ("cem", "hybrid"), ("hybrid", "noconcept")]
    out["paired"] = {}
    print("\nпарные разницы по повторам: точность (п.) [95%] | AP [95%]")
    for key in ("F", "A"):
        for kind in ("std", "unseen", "halves"):
            for a, b in pairs:
                fs = [f for f in FOLDS if f"{a}|{key}|{kind}|{f}" in out["rows"] and f"{b}|{key}|{kind}|{f}" in out["rows"]]
                if len(fs) < 2:
                    continue
                da = [100 * (out["rows"][f"{a}|{key}|{kind}|{f}"][0] - out["rows"][f"{b}|{key}|{kind}|{f}"][0]) for f in fs]
                dp = [out["rows"][f"{a}|{key}|{kind}|{f}"][1] - out["rows"][f"{b}|{key}|{kind}|{f}"][1] for f in fs]
                ia, ip = interval(da), interval(dp)
                out["paired"][f"{a}-{b}|{key}|{kind}"] = {"n": len(fs), "acc": ia, "ap": ip}
                print(f"  {key} {kind:6s} {a:9s} − {b:9s} n={len(fs)}: {ia[0]:+.2f} [{ia[1]:+.2f}; {ia[2]:+.2f}] | "
                      f"{ip[0]:+.4f} [{ip[1]:+.4f}; {ip[2]:+.4f}]", flush=True)
    out["phi"] = phi_analysis()
    print("\nсвязь разницы по атрибуту с max|φ| (Спирмен по повторам):", json.dumps(out["phi"], ensure_ascii=False))
    json.dump(out, open(os.path.join(R.OUT, "analysis_extra.json"), "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
