#!/usr/bin/env python
"""Читаемость отложенных концептов в зависимости от «бюджета» пробы (macro-AP на тесте):
  ridge_alpha -- ridge с одним alpha для всех концептов, alpha = mean(l) * {1e-4 .. 1e2} (от слабого штрафа к сильному);
  grad_epochs -- градиентная проба как прежний этап 2 (train.stage2: AdamW lr 1e-2, wd 0.01, BCE 0.5 / 10, батч 1024)
                 после 1, 3, 10, 30, 100 эпох;
  top_k       -- ridge (alpha по val для каждого концепта) только на первых k главных компонентах z (k = 4 .. все):
                 сколько информации о B лежит в направлениях с большой дисперсией;
  labels      -- ridge с alpha по 4-кратной кросс-валидации на n = 50 .. все меток fit (3 случайные выборки).
Гипотеза: схлопывание прячет B в направления с малой дисперсией -- разрыв между task и остальными велик при малом
бюджете и исчезает при сходимости. Результат: results/budget_<набор>.jsonl.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import train as T
from probe_eval import rescale, spec
from probe_ridge import ALPHAS, SUFFIX, ap_columns, ridge_path

HERE = Path(__file__).resolve().parent
EPOCHS = [1, 3, 10, 30, 100]
TOPK = [4, 16, 64, 256]
NLAB = [50, 200, 1000, 5000]


def macro(S, Y):
    cols = Y.sum(0) >= 5
    return float(ap_columns(S[:, cols], Y[:, cols]).mean())


def best_by_val(paths, Yv):
    """Для каждого концепта -- alpha с лучшим AP на валидации; оценки на тесте."""
    best, S = None, None
    for sv, st in paths:
        apv = ap_columns(sv, Yv).nan_to_num(0.0)
        if best is None:
            best, S = apv, st.clone()
        else:
            better = apv > best
            best = torch.where(better, apv, best)
            S[:, better] = st[:, better]
    return S


def grad_curve(X, Y, split, seed):
    """Прежняя проба этапа 2 с разным числом эпох (без выбора эпохи по валидации)."""
    gen = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    gate = T.concept_head(X.shape[1], Y.shape[1]).to(T.DEV)
    opt = torch.optim.AdamW(gate.parameters(), lr=1e-2, weight_decay=0.01)
    fit, te = split["fit"], split["test"]
    Xt, Yt = X[te].float(), Y[te].float()
    out = {}
    for ep in range(1, max(EPOCHS) + 1):
        for b in T.batches(fit, 1024, True, gen):
            loss = T.bce_concepts(gate(X[b.to(T.DEV)].float()), Y[b.to(T.DEV)].float())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        if ep in EPOCHS:
            with torch.no_grad():
                out[str(ep)] = macro(gate(Xt), Yt)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sets", nargs="+")
    ap.add_argument("--methods", nargs="+", default=["input", "task", "bce", "rank", "rand", "recon", "unif", "cbm", "cem"])
    args = ap.parse_args()
    for name in args.sets:
        ds, ev, emb = spec(name)
        zdir = HERE / "results" / "z" / name
        x, y, split, M = T.load(ds, ev, 0, emb, False, 0)
        if (zdir / "sub.npy").exists():
            sub = np.load(zdir / "sub.npy")
            x, y, M = x[sub], y[sub], M[sub]
            split = {"fit": torch.arange(12000), "val": torch.arange(12000, 14000), "test": torch.arange(14000, len(sub))}
        Y = torch.tensor(M.toarray(), dtype=torch.float64, device=T.DEV)
        out = HERE / "results" / f"budget{SUFFIX}_{name}.jsonl"
        done = {(r["method"], r["lambda"], r["seed"]) for r in map(json.loads, open(out))} if out.exists() else set()
        items = [("input", 0.0, 0, F.normalize(x, dim=-1))] if "input" in args.methods else []
        for f in sorted(zdir.glob("*_s*.npy")):
            method, lam, seed = f.stem.rsplit("_", 2)
            if method in args.methods and float(lam) == (3.0 if method == "recon" else 0.0 if method == "task" else 1.0):
                items.append((method, float(lam), int(seed[1:]), f))
        fit, val, te = (split[k].to(T.DEV) for k in ("fit", "val", "test"))
        for method, lam, seed, src in items:
            if (method, lam, seed) in done:
                continue
            Z = src if torch.is_tensor(src) else torch.tensor(np.load(src).astype(np.float32))
            X = rescale(Z.double(), split["fit"]).to(T.DEV)
            row = {"set": name, "method": method, "lambda": lam, "seed": seed}
            # 1) один alpha для всех концептов
            paths = ridge_path(X[fit], Y[fit], [X[val], X[te]])
            row["ridge_alpha"] = {f"{a:g}": macro(st, Y[te]) for a, (_, st) in zip(ALPHAS, paths)}
            row["ridge_cv"] = macro(best_by_val(paths, Y[val]), Y[te])
            # 2) градиентная проба с разным числом эпох
            row["grad_epochs"] = grad_curve(X, Y, {"fit": split["fit"], "test": split["test"]}, seed)
            # 3) первые k главных компонент
            mu = X[fit].mean(0)
            _, _, Vh = torch.linalg.svd(X[fit] - mu, full_matrices=False)
            row["top_k"] = {}
            for k in TOPK + [X.shape[1]]:
                if k > X.shape[1]:
                    continue
                Pk = (X - mu) @ Vh[:k].t()
                row["top_k"][str(k)] = macro(best_by_val(ridge_path(Pk[fit], Y[fit], [Pk[val], Pk[te]]), Y[val]), Y[te])
            # 4) число меток
            row["labels"] = {}
            for n in NLAB + [len(fit)]:
                if n > len(fit):
                    continue
                vals = []
                for rep in range(3 if n < len(fit) else 1):
                    g = torch.Generator().manual_seed(2000 + 10 * seed + rep)
                    few = split["fit"][torch.randperm(len(split["fit"]), generator=g)[:n]].to(T.DEV)
                    folds = torch.arange(len(few), device=T.DEV) % 4
                    cv = np.zeros(len(ALPHAS))
                    for f_ in range(4):
                        tr, va = few[folds != f_], few[folds == f_]
                        for i, (sv,) in enumerate(ridge_path(X[tr], Y[tr], [X[va]])):
                            cols = Y[va].sum(0) >= 2
                            cv[i] += float(ap_columns(sv[:, cols], Y[va][:, cols]).nan_to_num(0.0).mean())
                    st = ridge_path(X[few], Y[few], [X[te]])[int(cv.argmax())][0]
                    vals.append(macro(st, Y[te]))
                row["labels"][str(n)] = float(np.mean(vals))
            with open(out, "a") as fh:
                fh.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
