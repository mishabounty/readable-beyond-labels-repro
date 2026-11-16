#!/usr/bin/env python
"""Чтение B при известных A с отдельным масштабом блока A (план experiments/PLAN_2026-11-02_checks.md, N9).

То же, что conditional_probe.py (ridge, признаки [z, A]: каждый блок центрируется и делится на свой скаляр), но блок A
умножается на s ∈ {0.3, 1, 3, 10, 30}: штраф ridge на A слабее в s² раз. s выбирается для каждого z (модель × сид) по
средней AP на val при alpha, выбранном по val для каждого понятия (как в probe_ridge.probe); итог на test — probe_ridge.probe
при выбранном s. Сравнение по сидам: bce − rand и [z, A] − одни A у каждой модели.
Запуск: RIDGE_GRID=ext python cond_probe_scaled.py <наборы>; результат: results/cond_probe_scaled<SUFFIX>_<набор>.jsonl.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import train as T
from probe_eval import rescale, spec
from probe_ridge import ALPHAS, SUFFIX, ap_columns, probe, ridge_path

HERE = Path(__file__).resolve().parent
EXP_DATA = HERE / "data"
S_LIST = [0.3, 1.0, 3.0, 10.0, 30.0]


def val_score(X, Y, split):
    """Средняя AP на val при alpha, выбранном по val для каждого понятия."""
    dev = T.DEV if X.shape[1] <= 1024 else "cpu"
    fit, val = split["fit"].numpy(), split["val"].numpy()
    Xd = torch.tensor(X, device=dev, dtype=torch.float64)
    Yd = torch.tensor(Y, device=dev, dtype=torch.float64)
    best = torch.full((Y.shape[1],), -1.0, device=dev, dtype=torch.float64)
    for (sv,) in ridge_path(Xd[fit], Yd[fit], [Xd[val]]):
        best = torch.maximum(best, ap_columns(sv, Yd[val]).nan_to_num(-1.0))
    return float(best[best >= 0].mean())


def load_set(name):
    """Как в conditional_probe.py: данные, метки задачи, B, A и деление."""
    ds, ev, emb = spec(name)
    zdir = HERE / "results" / "z" / name
    x, y, split, M = T.load(ds, ev, 0, emb, False, 0)
    MA = T.load(ds, ev[:-1] + "A", 0, emb, False, 0)[3]
    if (zdir / "sub.npy").exists():
        sub = np.load(zdir / "sub.npy")
        if "_length" in name and ds == "goemotions":
            from text_finetune import length_labels
            y = length_labels()
        for task in ("country",):
            if f"_{task}" in name:
                y = torch.tensor(np.load(EXP_DATA / ds / f"labels_{task}.npy"), dtype=torch.long)
        x, y, M, MA = x[sub], y[sub], M[sub], MA[sub]
        nf, nv = (int(v) for v in np.load(zdir / "sub_split.npz").values()) if (zdir / "sub_split.npz").exists() \
            else (12000, 2000)
        split = {"fit": torch.arange(nf), "val": torch.arange(nf, nf + nv), "test": torch.arange(nf + nv, len(sub))}
    return zdir, x, y, split, M.toarray().astype(np.float32), MA


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sets", nargs="+")
    args = ap.parse_args()
    for name in args.sets:
        zdir, x, y, split, Y, MA = load_set(name)
        fit = split["fit"]
        A = rescale(torch.tensor(MA.toarray(), dtype=torch.float64), fit)
        out = HERE / "results" / f"cond_probe_scaled{SUFFIX}_{name}.jsonl"
        done = {(r["method"], r["lambda"], r["seed"]) for r in map(json.loads, open(out))} if out.exists() else set()
        items = [("concepts_A", 0.0, 0, None), ("input", 0.0, 0, rescale(F.normalize(x, dim=-1).double(), fit))]
        for f in sorted(zdir.glob("*_s*.npy")):
            method, lam, seed = f.stem.rsplit("_", 2)
            items.append((method, float(lam), int(seed[1:]), f))
        for method, lam, seed, src in items:
            if (method, lam, seed) in done:
                continue
            if method == "concepts_A":
                m = probe(A.float().numpy(), Y, split, y, seed)
                row = {"set": name, "method": method, "lambda": lam, "seed": seed, "s": None, **m}
            else:
                Z = src if not isinstance(src, Path) else rescale(torch.tensor(np.load(src).astype(np.float64)), fit)
                scores = {s: val_score(torch.cat([Z, s * A], 1).float().numpy(), Y, split) for s in S_LIST}
                s_best = max(S_LIST, key=lambda s: scores[s])
                m = probe(torch.cat([Z, s_best * A], 1).float().numpy(), Y, split, y, seed)
                row = {"set": name, "method": method, "lambda": lam, "seed": seed, "s": s_best,
                       "val_by_s": scores, **m}
            with open(out, "a") as fh:
                fh.write(json.dumps(row) + "\n")
            print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()
                              if k not in ("val_by_s",)}), flush=True)
        # сводка по набору
        g = defaultdict(dict)
        for r in map(json.loads, open(out)):
            g[r["method"]][r["seed"]] = r["macro_ap"]
        a0 = np.mean(list(g["concepts_A"].values()))
        print(f"\n{name}: одни A {a0:.4f}; [z, A] − A: " + ", ".join(
            f"{m} {np.mean(list(v.values())) - a0:+.4f}" for m, v in g.items() if m != "concepts_A"), flush=True)
        if "bce" in g and "rand" in g:
            ks = sorted(set(g["bce"]) & set(g["rand"]))
            d = [g["bce"][k] - g["rand"][k] for k in ks]
            rng = np.random.RandomState(0)
            bs = [np.mean(rng.choice(list(g["bce"].values()), len(g["bce"]))) -
                  np.mean(rng.choice(list(g["rand"].values()), len(g["rand"]))) for _ in range(5000)]
            print(f"{name}: bce − rand на [z, sA]: {np.mean(d):+.4f} по сидам {np.round(d, 4).tolist()}; "
                  f"бутстрэп по сидам [{np.percentile(bs, 2.5):+.4f}; {np.percentile(bs, 97.5):+.4f}]", flush=True)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
