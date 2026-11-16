#!/usr/bin/env python
"""Чтение отложенных атрибутов B (cemB, атрибуты вида) для быстрой проверки опыта CEM §5.3 (cem_check.py).

Для каждого представления (исходные признаки ResNet-18; z и узкое место обученных моделей) две меры:
  мера CEM -- логистическая регрессия на каждый атрибут B (признаки стандартизованы по fit, C подобран на val
             из сетки; также C = 1 без подбора), средняя точность на тесте; рядом -- тривиальный уровень
             (ответ большинства по fit);
  наша     -- macro-AP ridge до конца (probe_ridge.probe, сетка штрафа ext).
Запуск без GPU: CUDA_VISIBLE_DEVICES= (ridge в float64 на CPU). Итог: results/cem_readout.json и таблица в stdout.
"""

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression

os.environ.setdefault("RIDGE_GRID", "ext")
import probe_ridge as PR  # noqa: E402
import train as T  # noqa: E402
from probe_eval import rescale  # noqa: E402

HERE = Path(__file__).resolve().parent
ZDIR = HERE / "results" / "z" / "cem_check"
CS = [0.01, 0.1, 1.0, 10.0]


def logreg_acc(X, Y, fit, val, te):
    """Средняя по атрибутам точность на тесте: C подобран на val и C = 1."""
    mu, sd = X[fit].mean(0), X[fit].std(0) + 1e-6
    X = (X - mu) / sd

    def one(j):
        y = Y[:, j]
        if y[fit].min() == y[fit].max():
            acc = float((y[te] == y[fit][0]).mean())
            return acc, acc
        res = {}
        for c in CS:
            m = LogisticRegression(C=c, max_iter=1000).fit(X[fit], y[fit])
            res[c] = (float((m.predict(X[val]) == y[val]).mean()), float((m.predict(X[te]) == y[te]).mean()))
        best = max(CS, key=lambda c: (res[c][0], -abs(np.log10(c))))
        return res[best][1], res[1.0][1]

    out = Parallel(n_jobs=8)(delayed(one)(j) for j in range(Y.shape[1]))
    return float(np.mean([o[0] for o in out])), float(np.mean([o[1] for o in out]))


def paired(rows, a, b, key):
    """Разница a - b по запускам (одинаковые сиды): среднее, 95% t-интервал, число положительных."""
    from scipy import stats
    pat = re.compile(r"(.+)_s(\d+)_([zb])$")
    by = defaultdict(dict)
    for name, r in rows.items():
        m = pat.match(name)
        if m:
            by[(m.group(1), m.group(3))][int(m.group(2))] = r[key]
    seeds = sorted(set(by[a]) & set(by[b]))
    d = np.array([by[a][s_] - by[b][s_] for s_ in seeds])
    if len(d) < 2:
        return None
    h = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    return {"mean": float(d.mean()), "lo": float(d.mean() - h), "hi": float(d.mean() + h),
            "pos": int((d > 0).sum()), "n": len(d)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    global ZDIR
    ZDIR = HERE / "results" / "z" / f"cem_check{args.tag}"
    x, y, split, M = T.load("cub", "cemB", 0, "resnet18", False, 0)
    Y = M.toarray().astype(np.int64)
    fit, val, te = (split[k].numpy() for k in ("fit", "val", "test"))
    maj = Y[fit].mean(0) >= 0.5
    trivial = float(np.mean([(Y[te, j] == maj[j]).mean() for j in range(Y.shape[1])]))
    reps = {"исходная ResNet-18, layer4": x.numpy()}
    l3 = np.load(HERE / "data" / "cub" / "resnet18_layer3.npy", mmap_mode="r")
    reps["исходная ResNet-18, layer3"] = np.concatenate(
        [np.asarray(l3[i:i + 512], dtype=np.float32).mean((2, 3)) for i in range(0, len(l3), 512)])
    for f in sorted(ZDIR.glob("*_s*_[zb].npy")):
        reps[f.stem] = np.load(f)
    rows = {}
    for name, R in reps.items():
        acc_tuned, acc_c1 = logreg_acc(R, Y, fit, val, te)
        Xr = rescale(torch.tensor(R, dtype=torch.float64), split["fit"]).float().numpy()
        ap = PR.probe(Xr, Y.astype(np.float32), split, y, 0)["macro_ap"]
        rows[name] = {"acc": acc_tuned, "acc_c1": acc_c1, "macro_ap": float(ap), "dim": int(R.shape[1])}
        print(f"{name:42s} dim={R.shape[1]:4d}  acc={acc_tuned:.4f}  acc(C=1)={acc_c1:.4f}  AP={ap:.4f}", flush=True)
    # сводка по запускам: метод_концепты_sK_{z,b}
    agg = defaultdict(list)
    for name, r in rows.items():
        m = re.match(r"(.+)_s\d+_([zb])$", name)
        agg[(m.group(1), m.group(2)) if m else (name, "")].append(r)
    tr = HERE / "results" / f"cem_check{args.tag}.jsonl"
    train_rows = [json.loads(l) for l in open(tr)] if tr.exists() else []
    print(f"\nтривиальный уровень (ответ большинства): точность {trivial:.4f}")
    print("представление | точность (C подобран) | AP | точность задачи (тест)")
    for (k, part), rs in agg.items():
        accs, aps = [r["acc"] for r in rs], [r["macro_ap"] for r in rs]
        t_acc = [r["test_acc"] for r in train_rows if r.get("tag", f"{r['method']}_{r['concepts']}") == k]
        print(f"{k} {part} | {np.mean(accs):.4f} [{min(accs):.4f}; {max(accs):.4f}] | {np.mean(aps):.4f} "
              f"[{min(aps):.4f}; {max(aps):.4f}] | {np.mean(t_acc):.3f}" if t_acc else
              f"{k} {part} | {np.mean(accs):.4f} | {np.mean(aps):.4f} | —")
    comps = {}
    print("\nпарные разницы по запускам: среднее [95% t-интервал] (положительных из n)")
    for a, b in ((("cem0.1_cemA", "b"), ("cem0.1_cemArnd", "b")), (("cem0.1_cemA", "z"), ("cem0.1_cemArnd", "z")),
                 (("cem0.1_cemA", "b"), ("cem0_cemA", "b")), (("cem0.1_cemArnd", "b"), ("cem0_cemA", "b")),
                 (("task0_ep2_cemA", "z"), ("cem0.1_cemA", "b")), (("task0_cemA", "z"), ("cem0.1_cemA", "b")),
                 (("hcbm0.1_cemA", "b"), ("cem0.1_cemA", "b"))):
        for key in ("acc", "macro_ap"):
            r = paired(rows, a, b, key)
            if r:
                comps[f"{a[0]}_{a[1]} - {b[0]}_{b[1]} [{key}]"] = r
                print(f"{a[0]} {a[1]} - {b[0]} {b[1]} [{key}]: {r['mean']:+.4f} [{r['lo']:+.4f}; {r['hi']:+.4f}] "
                      f"({r['pos']} из {r['n']})")
    json.dump({"trivial_acc": trivial, "rows": rows, "paired": comps},
              open(HERE / "results" / f"cem_readout{args.tag}.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
