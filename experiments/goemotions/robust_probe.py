#!/usr/bin/env python
"""Проверки пробы для наборов z обучаемых энкодеров (results/z/<набор>): рецензия v2, замечания M4, M7, минорное 9.

  mlp    -- нелинейная проба: MLP (512, ReLU, dropout 0.1), BCE, AdamW, ранняя остановка по macro-AP на val;
            отвечает, потеряна ли информация о B или лишь закодирована нелинейно;
  mlpw   -- та же MLP на отбеленных признаках: у схлопнутого z информация лежит в направлениях с малой
            дисперсией, и градиентная проба без отбеливания до них не доходит (тот же эффект бюджета, что у линейной);
  fresh  -- ridge-проба на примерах, которых этап 1 не видел в градиентах: val и test этапа 1 объединяются и
            3 раза случайно делятся 60 / 15 / 25 (fit / val / test пробы); снимает сдвиг train/test у z;
  xz     -- ridge на [x, z] (вход до обучения и z, каждый блок со своим масштабом): добавляет ли z что-то сверх x;
  boot   -- иерархический бутстрап разностей macro-AP: сиды и тестовые примеры ресемплируются вместе
            (оценки ridge без Платта; Платт монотонен и не меняет AP отдельного концепта).
Результат: results/robust_<набор>.jsonl (mlp / fresh / xz по файлам z) и results/boot_<набор>.json.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

import train as T
from probe_eval import rescale, spec, whiten
from probe_ridge import ALPHAS, SUFFIX, ap_columns, probe, ridge_path

HERE = Path(__file__).resolve().parent
R = HERE / "results"
PAIRS = [("bce", "rand"), ("bce", "input"), ("task", "input"), ("bce", "aonly"), ("bce", "km"), ("bce", "hyp"),
         ("bce", "recon"), ("bce", "unif"), ("rand", "task"), ("bce", "wise"), ("bce", "lpft"), ("bce", "l2sp1")]


def load_set(name):
    ds, ev, emb = spec(name)
    zdir = R / "z" / name
    x, y, split, M = T.load(ds, ev, 0, emb, False, 0)
    if (zdir / "sub.npy").exists():
        sub = np.load(zdir / "sub.npy")
        x, y, M = x[sub], y[sub], M[sub]
        nf, nv = (int(v) for v in np.load(zdir / "sub_split.npz").values()) if (zdir / "sub_split.npz").exists() \
            else (12000, 2000)
        split = {"fit": torch.arange(nf), "val": torch.arange(nf, nf + nv), "test": torch.arange(nf + nv, len(sub))}
    items = [("input", 0.0, 0, F.normalize(x, dim=-1))]
    for f in sorted(zdir.glob("*_s*.npy")):
        method, lam, seed = f.stem.rsplit("_", 2)
        items.append((method, float(lam), int(seed[1:]), f))
    return x, y, split, M.toarray().astype(np.float32), items


def mlp_probe(X, Y, split, seed, hidden=512, epochs=100, patience=8):
    torch.manual_seed(seed)
    Xd, Yd = torch.tensor(X, device=T.DEV), torch.tensor(Y, device=T.DEV)
    net = nn.Sequential(nn.Linear(X.shape[1], hidden), nn.ReLU(), nn.Dropout(0.1), nn.Linear(hidden, Y.shape[1])).to(T.DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    fit, val, te = (split[k].to(T.DEV) for k in ("fit", "val", "test"))
    g = torch.Generator(device="cpu").manual_seed(seed)
    best, stale = None, 0
    for ep in range(epochs):
        net.train()
        perm = fit[torch.randperm(len(fit), generator=g).to(T.DEV)]
        for i in range(0, len(perm), 512):
            b = perm[i:i + 512]
            loss = F.binary_cross_entropy_with_logits(net(Xd[b]), Yd[b])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            apv = float(ap_columns(net(Xd[val]).double(), Yd[val].double()).nanmean())
            if best is None or apv > best[0] + 1e-4:
                best, stale = (apv, net(Xd[te]).double()), 0
            else:
                stale += 1
                if stale >= patience:
                    break
    return float(ap_columns(best[1], Yd[te].double()).nanmean())


def ridge_scores(X, Y, split):
    """Оценки ridge на тесте (alpha по val AP для каждого концепта), без калибровки."""
    dev = T.DEV if X.shape[1] <= 1024 else "cpu"
    Xd, Yd = torch.tensor(X, device=dev, dtype=torch.float64), torch.tensor(Y, device=dev, dtype=torch.float64)
    fit, val, te = (split[k].numpy() for k in ("fit", "val", "test"))
    paths = ridge_path(Xd[fit], Yd[fit], [Xd[val], Xd[te]])
    best = torch.full((Y.shape[1],), -1.0, device=dev, dtype=torch.float64)
    S = torch.zeros(len(te), Y.shape[1], device=dev, dtype=torch.float64)
    for _, (sv, st) in zip(ALPHAS, paths):
        apv = ap_columns(sv, Yd[val]).nan_to_num(0.0)
        better = apv > best
        best[better] = apv[better]
        S[:, better] = st[:, better]
    return S.float().cpu()


def boot_macro_ap(S, Y, idx):
    """macro-AP по концептам с >= 5 положительными на полном тесте, для набора ресемплов idx (B x n)."""
    out = []
    for b in range(len(idx)):
        Sb, Yb = S[idx[b]], Y[idx[b]]
        out.append(float(ap_columns(Sb, Yb).nanmean()))
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sets", nargs="+")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--skip", nargs="*", default=[], help="пропустить проверки: mlp fresh xz boot")
    args = ap.parse_args()
    for name in args.sets:
        x, y, split, Y, items = load_set(name)
        out = R / f"robust{SUFFIX}_{name}.jsonl"
        done = {(r["method"], r["lambda"], r["seed"], r["check"]) for r in map(json.loads, open(out))} \
            if out.exists() else set()
        pool = torch.cat([split["val"], split["test"]])
        xin = rescale(F.normalize(x, dim=-1).double(), split["fit"])
        te = split["test"]
        Yte = torch.tensor(Y[te.numpy()])
        keep = Yte.sum(0) >= 5
        scores = {}
        for method, lam, seed, src in items:
            Z = src if torch.is_tensor(src) else torch.tensor(np.load(src).astype(np.float32))
            X = rescale(Z.double(), split["fit"]).float().numpy()
            rows = []
            if "mlp" not in args.skip and (method, lam, seed, "mlp") not in done:
                rows.append(("mlp", mlp_probe(X, Y, split, seed)))
            if "mlpw" not in args.skip and (method, lam, seed, "mlpw") not in done:
                Xw = whiten(Z.double(), split["fit"]).float().numpy()
                rows.append(("mlpw", mlp_probe(Xw, Y, split, seed)))
            if "fresh" not in args.skip and (method, lam, seed, "fresh") not in done:
                vals = []
                for rep in range(3):
                    g = torch.Generator().manual_seed(100 + rep)
                    p = pool[torch.randperm(len(pool), generator=g)]
                    n1, n2 = int(0.6 * len(p)), int(0.75 * len(p))
                    sp_ = {"fit": p[:n1].sort().values, "val": p[n1:n2].sort().values, "test": p[n2:].sort().values}
                    Xf = rescale(Z.double(), sp_["fit"]).float().numpy()
                    vals.append(probe(Xf, Y, sp_, y, seed)["macro_ap"])
                rows.append(("fresh", float(np.mean(vals))))
            if "xz" not in args.skip and method != "input" and (method, lam, seed, "xz") not in done:
                Xxz = torch.cat([xin, torch.tensor(X, dtype=torch.float64)], 1).float().numpy()
                rows.append(("xz", probe(Xxz, Y, split, y, seed)["macro_ap"]))
            for check, v in rows:
                row = {"set": name, "method": method, "lambda": lam, "seed": seed, "check": check, "macro_ap": v}
                with open(out, "a") as fh:
                    fh.write(json.dumps(row) + "\n")
                print(json.dumps(row), flush=True)
            if "boot" not in args.skip:
                # ключ с lambda для L2-SP (раньше три штрафа затирали друг друга)
                key = f"{method}{lam:g}" if method == "l2sp" else method
                scores.setdefault(key, {})[seed] = ridge_scores(X, Y, split)[:, keep]
        if "boot" in args.skip:
            continue
        Yk = Yte[:, keep]
        rng = np.random.default_rng(0)
        res = {}
        for a, b in PAIRS:
            if a not in scores or b not in scores:
                continue
            sa, sb = scores[a], scores[b]
            seeds_a = sorted(sa)
            diffs = []
            for _ in range(args.boot):
                idx = torch.tensor(rng.integers(0, len(Yk), len(Yk)))
                # сиды каждого метода ресемплируются независимо (раньше недостающие сиды b заменялись сидом 0)
                ss = rng.choice(seeds_a, len(seeds_a))
                sb_ = rng.choice(sorted(sb), len(sb))
                ma = np.mean([boot_macro_ap(sa[s], Yk, idx[None])[0] for s in ss])
                mb = np.mean([boot_macro_ap(sb[s], Yk, idx[None])[0] for s in sb_])
                diffs.append(ma - mb)
            full = torch.arange(len(Yk))[None]
            pa = np.mean([boot_macro_ap(sa[s], Yk, full)[0] for s in seeds_a])
            pb = np.mean([boot_macro_ap(sb[s], Yk, full)[0] for s in sorted(sb)])
            res[f"{a}-{b}"] = {"diff": float(pa - pb), "lo": float(np.percentile(diffs, 2.5)),
                               "hi": float(np.percentile(diffs, 97.5)), "seeds_a": len(sa), "seeds_b": len(sb)}
            print(name, a, b, res[f"{a}-{b}"], flush=True)
        (R / f"boot{SUFFIX}_{name}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
