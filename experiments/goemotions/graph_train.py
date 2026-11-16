#!/usr/bin/env python
"""Графы без GFT: двухслойная GraphSAGE поверх эмбеддингов текстов узлов (data/graph_<имя>/, graph_prepare.py).

z = normalize(SAGE2(relu(SAGE1(x)))) (256-d), линейная голова задачи (логиты x10), вспомогательные цели как в
текстовых постановках: bce -- BCE на словах A; rand -- то же на случайных концептах той же плотности; unif --
равномерность (на 2048 случайных узлах fit за шаг); recon -- восстановление x из z (1 - cos). Полный батч, AdamW,
ранняя остановка по macro-F1 на валидации. Задачи: topic (классы узлов), degree (квартиль степени).
Читаемость отложенных слов B -- ridge-проба до сходимости (probe_ridge.probe); ориентиры: вход x, [x, среднее x
соседей], метка задачи (связанность); условная проба [z, метки A].
Результат: results/graph_<имя>_<задача>.jsonl; уже посчитанные строки пропускаются.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import scatter

import train as T
from collapse_check import collapse_metrics
from probe_eval import rescale
from probe_ridge import probe

HERE = Path(__file__).resolve().parent


class GNN(nn.Module):
    def __init__(self, d_in, d=256):
        super().__init__()
        self.c1, self.c2 = SAGEConv(d_in, d), SAGEConv(d, d)

    def forward(self, x, ei):
        h = F.dropout(F.relu(self.c1(x, ei)), 0.1, self.training)
        return F.normalize(self.c2(h, ei), dim=-1)


def randomize(M, seed):
    rng = np.random.default_rng(10_000 + seed)
    counts = np.diff(M.indptr)
    rows = np.repeat(np.arange(M.shape[0]), counts)
    cols = np.concatenate([rng.choice(M.shape[1], c, replace=False) for c in counts if c] or [np.array([], int)])
    return sp.csr_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=M.shape)


def train(x, ei, y, split, MA, method, lam, seed, epochs=300, patience=30):
    torch.manual_seed(seed)
    n_cls = int(y.max()) + 1
    enc, head = GNN(x.shape[1]).to(T.DEV), nn.Linear(256, n_cls).to(T.DEV)
    aux = {"bce": lambda: T.concept_head(256, MA.shape[1]), "rand": lambda: T.concept_head(256, MA.shape[1]),
           "recon": lambda: nn.Linear(256, x.shape[1])}.get(method, lambda: None)()
    groups = [{"params": list(enc.parameters()) + list(head.parameters()), "lr": 5e-3}]
    if aux is not None:
        aux = aux.to(T.DEV)
        groups.append({"params": aux.parameters(), "lr": 1e-2})
    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    fit, val = split["fit"].to(T.DEV), split["val"].to(T.DEV)
    yd = y.to(T.DEV)
    Ma = torch.tensor(MA.toarray(), dtype=torch.float32, device=T.DEV) if method in ("bce", "rand") else None
    g = torch.Generator(device="cpu").manual_seed(seed)
    best, stale = None, 0
    for ep in range(epochs):
        enc.train()
        z = enc(x, ei)
        loss = F.cross_entropy(10.0 * head(z[fit]), yd[fit])
        if method in ("bce", "rand"):
            loss = loss + lam * T.bce_concepts(aux(z[fit]), Ma[fit])
        elif method == "recon":
            loss = loss + lam * (1 - F.cosine_similarity(aux(z[fit]), x[fit], dim=-1)).mean()
        elif method == "unif":
            b = fit[torch.randperm(len(fit), generator=g)[:2048].to(T.DEV)]
            zb = z[b]  # |zi - zj|^2 = 2 - 2 zi.zj (z нормирован): через матрицу Грама, pdist в backward требует n*n*d
            off = ~torch.eye(len(zb), dtype=torch.bool, device=zb.device)
            loss = loss + lam * (4 * (zb @ zb.t()) - 4)[off].exp().mean().log()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        enc.eval()
        with torch.no_grad():
            zv = enc(x, ei)
            f1 = T.macro_f1(y[split["val"]], (10.0 * head(zv[val])).argmax(1).cpu(), n_cls)
        if best is None or f1 > best[0] + 1e-4:
            best, stale = (f1, ep, zv.cpu(), head.state_dict()), 0
        else:
            stale += 1
            if stale >= patience:
                break
    head.load_state_dict(best[3])
    te = split["test"]
    with torch.no_grad():
        pt = (10.0 * head(best[2][te].to(T.DEV))).argmax(1).cpu()
    return best[2], {"best_epoch": best[1], "task_f1": T.macro_f1(y[te], pt, n_cls)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("graphs", nargs="+")
    ap.add_argument("--task", default="topic", choices=["topic", "degree"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--runs", nargs="+", default=["task:0", "bce:1", "rand:1", "unif:1", "recon:3"])
    args = ap.parse_args()
    for gname in args.graphs:
        d = HERE / "data" / f"graph_{gname}"
        x = torch.tensor(np.load(d / "emb_mqa.npy").astype(np.float32), device=T.DEV)
        ei = torch.tensor(np.load(d / "edge_index.npy"), device=T.DEV)
        y = torch.tensor(np.load(d / f"labels_{args.task}.npy"))
        s = np.load(d / "split.npz")
        split = {k: torch.tensor(s[k]) for k in ("fit", "val", "test")}
        MA, MB = (sp.load_npz(d / f"concepts_word{h}.npz").tocsr() for h in "AB")
        YB = MB.toarray().astype(np.float32)
        A = rescale(torch.tensor(MA.toarray(), dtype=torch.float64), split["fit"])
        out = HERE / "results" / f"graph_{gname}_{args.task}.jsonl"
        done = {(r["method"], r["seed"], r["features"]) for r in map(json.loads, open(out))} if out.exists() else set()

        def emit(method, seed, feats, Z, extra=None):
            if (method, seed, feats) in done:
                return
            X = rescale(Z.double().cpu(), split["fit"])
            if feats == "z+A":
                X = torch.cat([X, A], 1)
            m = probe(X.float().numpy(), YB, split, y, seed)
            row = {"graph": gname, "task": args.task, "method": method, "seed": seed, "features": feats, **m,
                   **(extra or {})}
            with open(out, "a") as fh:
                fh.write(json.dumps(row) + "\n")
            print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)

        xn = F.normalize(x, dim=-1)
        nb = F.normalize(scatter(x[ei[0]], ei[1], 0, dim_size=len(x), reduce="mean"), dim=-1)
        emit("input", 0, "z", xn)
        emit("input_neighbors", 0, "z", torch.cat([xn, nb], 1))
        emit("task_label", 0, "z", F.one_hot(y).float())
        emit("input", 0, "z+A", xn)
        for seed in args.seeds:
            MR = randomize(MA, seed)
            for r in args.runs:
                method, lam = r.split(":")[0], float(r.split(":")[1])
                if (method, seed, "z") in done and (method, seed, "z+A") in done:
                    continue
                t0 = time.time()
                z, tm = train(x, ei, y, split, MR if method == "rand" else MA, method, lam, seed)
                extra = {"lambda": lam, **tm, **collapse_metrics(z[split["test"]], y[split["test"]]),
                         "seconds": time.time() - t0}
                emit(method, seed, "z", z, extra)
                emit(method, seed, "z+A", z, {"lambda": lam})
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
