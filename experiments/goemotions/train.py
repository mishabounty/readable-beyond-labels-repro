#!/usr/bin/env python
"""Готовит ли ранговая концептная потеря пространство z для последующего восстановления концептов?

Этап 1 (обучение представления), вход -- замороженный эмбеддинг текста x:
    z = normalize(MLP(x)),  классификатор задачи -- линейный по z;
    method = task : L = CE
    method = bce  : L = CE + lambda * BCE-концептов (присутствующие и отсутствующие усредняются отдельно, 0.5 / 10)
    method = rank : L = CE + lambda * ранговая потеря (присутствующий концепт выше отсутствующего, внутри текста
                    и внутри концепта; пары выбираются случайно)
    вспомогательная концептная голова после этапа 1 отбрасывается; ранняя остановка по val macro-F1 задачи.
Этап 2 (одинаковый для всех): z замораживается, новый линейный декодер концептов обучается BCE (0.5 / 10);
    эпоха выбирается по val AP. Отдельно тот же декодер обучается на сыром x (ориентир «без энкодера»).
Метрики на test: macro-F1 / accuracy задачи; AP (по всем парам текст--концепт), macro-AP (по концептам), полнота при
    точности >= 0.7, precision@5 (доля верных среди 5 лучших концептов текста), P/R и число концептов при пороге 0.5.
Результат каждого запуска дописывается строкой в results/<датасет>.jsonl.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from torch import nn

HERE = Path(__file__).resolve().parent
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------------------------------------------ данные
def load(dataset, dictionary, k, emb, random_concepts, seed):
    d = HERE / "data" / dataset
    x = torch.tensor(np.load(d / f"emb_{emb}.npy").astype(np.float32))
    y = torch.tensor(np.load(d / "labels.npy"), dtype=torch.long)
    s = np.load(d / "split.npz")
    M = sp.load_npz(d / f"concepts_{dictionary}.npz").tocsr()
    if k:
        M = M[:, :k].tocsr()
    if random_concepts:  # случайные координаты с тем же числом концептов у каждого текста
        rng = np.random.default_rng(10_000 + seed)
        counts = np.diff(M.indptr)
        rows = np.repeat(np.arange(M.shape[0]), counts)
        cols = np.concatenate([rng.choice(M.shape[1], c, replace=False) for c in counts if c] or [np.array([], int)])
        M = sp.csr_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=M.shape)
    return x, y, {k_: torch.tensor(s[k_]) for k_ in ("fit", "val", "test")}, M


def dense(M, idx):
    return torch.tensor(M[idx.numpy()].toarray(), dtype=torch.float32)


# ------------------------------------------------------------------ модель и потери
class Encoder(nn.Module):
    def __init__(self, dim, hidden=768, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, hidden))

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


def concept_head(dim, m):
    head = nn.Linear(dim, m)
    nn.init.normal_(head.weight, std=0.01)
    nn.init.constant_(head.bias, -1.0)
    return head


def bce_concepts(logits, target, w_pos=0.5, w_neg=10.0):
    """BCE, в которой присутствующие и отсутствующие концепты усредняются отдельно (как на этапе 2 для Cora)."""
    p = torch.sigmoid(logits).clamp(1e-6, 1 - 1e-6)
    pos = -(target * p.log()).sum() / target.sum().clamp_min(1)
    neg = -((1 - target) * torch.log1p(-p)).sum() / (1 - target).sum().clamp_min(1)
    return w_pos * pos + w_neg * neg


def concept_loss(method, logits, target):
    """Концептная потеря этапа 1: bce -- веса 0.5 / 10 (как на этапе 2 для Cora), bce_bal -- присутствующие и
    отсутствующие с равными весами, bce_plain -- обычная BCE (среднее по всем парам), rank -- ранжирование."""
    if method == "bce":
        return bce_concepts(logits, target)
    if method == "bce_bal":
        return bce_concepts(logits, target, 0.5, 0.5)
    if method == "bce_plain":
        return F.binary_cross_entropy_with_logits(logits, target)
    return rank_concepts(logits, target)


def vicreg_loss(z):
    """Дисперсия + ковариация VICReg (Bardes et al., 2022) без инвариантности (аугментаций нет); z нормирован,
    поэтому координаты домножаются на sqrt(d), чтобы цель std = 1 соответствовала изотропному z; веса 25 : 1."""
    zs = z * z.shape[1] ** 0.5
    zs = zs - zs.mean(0)
    var = F.relu(1 - (zs.var(0) + 1e-4).sqrt()).mean()
    cov = zs.t() @ zs / (len(zs) - 1)
    off = cov.pow(2).sum() - cov.diagonal().pow(2).sum()
    return var + off / zs.shape[1] / 25.0


def pairwise_rank(scores, target, margin=1.0, n_pos=32, n_neg=128):
    """mean softplus(m - (s_pos - s_neg)) по случайным парам «есть / нет» в каждой строке."""
    rows, cols = target.shape
    pos_idx = (torch.rand_like(target) + 2 * target).topk(min(n_pos, cols), dim=1).indices
    pos_ok = target.gather(1, pos_idx) > 0.5
    neg_idx = torch.randint(cols, (rows, n_neg), device=target.device)
    neg_ok = target.gather(1, neg_idx) < 0.5
    diff = scores.gather(1, pos_idx).unsqueeze(2) - scores.gather(1, neg_idx).unsqueeze(1)
    pairs = (pos_ok.unsqueeze(2) & neg_ok.unsqueeze(1)).float()
    n = pairs.sum(dim=(1, 2))
    per_row = (F.softplus(margin - diff) * pairs).sum(dim=(1, 2)) / n.clamp_min(1)
    keep = n > 0
    return per_row[keep].mean() if keep.any() else scores.sum() * 0


def rank_concepts(logits, target):
    """Внутри текста (концепты текста выше чужих) + внутри концепта (тексты с концептом выше текстов без него)."""
    return pairwise_rank(logits, target) + pairwise_rank(logits.t().contiguous(), target.t().contiguous())


# ------------------------------------------------------------------ метрики
def macro_f1(y, pred, c):
    f = []
    for k in range(c):
        tp = ((y == k) & (pred == k)).sum().item()
        fp = ((y != k) & (pred == k)).sum().item()
        fn = ((y == k) & (pred != k)).sum().item()
        f.append(0.0 if 2 * tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f))


@torch.no_grad()
def concept_metrics(scores, target):
    """scores, target: [тексты x концепты] на одном устройстве."""
    s, t = scores.flatten(), target.flatten()
    order = s.argsort(descending=True)
    tp = t[order].cumsum(0)
    prec = tp / torch.arange(1, len(tp) + 1, device=tp.device)
    rec = tp / t.sum().clamp_min(1)
    ap = float((prec * t[order]).sum() / t.sum().clamp_min(1))
    ok = prec >= 0.7
    r_p70 = float(rec[ok].max()) if ok.any() else 0.0
    # macro-AP по концептам, у которых в тесте >= 5 положительных
    cols = target.sum(0) >= 5
    S, T = scores[:, cols], target[:, cols]
    o = S.argsort(dim=0, descending=True)
    Ts = T.gather(0, o)
    cp = Ts.cumsum(0) / torch.arange(1, Ts.shape[0] + 1, device=Ts.device).unsqueeze(1)
    macro_ap = float(((cp * Ts).sum(0) / Ts.sum(0)).mean()) if cols.any() else 0.0
    # precision@5 по текстам, у которых есть хотя бы один концепт
    has = target.sum(1) > 0
    top = scores[has].topk(min(5, scores.shape[1]), dim=1).indices
    p_at5 = float(target[has].gather(1, top).mean())
    pred = torch.sigmoid(scores) >= 0.5
    inside = (pred & (target > 0.5)).sum().item()
    total = pred.sum().item()
    return {"ap": ap, "macro_ap": macro_ap, "recall_at_p70": r_p70, "p_at5": p_at5,
            "precision@0.5": inside / max(total, 1), "recall@0.5": inside / max(target.sum().item(), 1),
            "active_per_text": total / target.shape[0], "true_per_text": target.sum().item() / target.shape[0]}


# ------------------------------------------------------------------ этапы
def batches(idx, bs, shuffle, gen=None):
    if shuffle:
        idx = idx[torch.randperm(len(idx), generator=gen)]
    for i in range(0, len(idx), bs):
        yield idx[i:i + bs]


DEFAULT_KNOBS = {"logit_scale": 10.0, "label_smoothing": 0.0, "wd": 0.01, "fixed_epochs": 0}


def stage1(x, y, split, M, method, lam, args, gen, aux_head=None, recon_lam=0.0):
    """aux_head -- своя концептная голова вместо линейной (например, привязанная к названиям концептов);
    recon_lam > 0 -- дополнительно к концептной потере восстановление x из z (1 - cos) с этим весом.
    Необязательные поля args (ручки схлопывания при неизменной задаче): logit_scale (10), label_smoothing (0),
    wd (0.01), fixed_epochs (0 -- ранняя остановка по val F1; > 0 -- ровно столько эпох, берётся последняя)."""
    scale, smooth = getattr(args, "logit_scale", 10.0), getattr(args, "label_smoothing", 0.0)
    wd, fixed = getattr(args, "wd", 0.01), getattr(args, "fixed_epochs", 0)
    n_cls = int(y.max()) + 1
    enc, task = Encoder(x.shape[1]).to(DEV), nn.Linear(768, n_cls).to(DEV)
    if method == "recon":  # контроль без концептов: восстановление самого эмбеддинга x из z
        aux = nn.Linear(768, x.shape[1]).to(DEV)
    elif aux_head is not None:
        aux = aux_head.to(DEV)
    else:
        aux = concept_head(768, M.shape[1]).to(DEV) if method not in ("task", "unif", "vicreg") else None
    rec = nn.Linear(768, x.shape[1]).to(DEV) if recon_lam > 0 else None
    groups = [{"params": list(enc.parameters()) + list(task.parameters()), "lr": args.lr}]
    if aux is not None:
        groups.append({"params": aux.parameters(), "lr": 1e-2})
    if rec is not None:
        groups.append({"params": rec.parameters(), "lr": 1e-2})
    opt = torch.optim.AdamW(groups, weight_decay=wd)
    best, stale, hist = None, 0, []
    for ep in range(1, (fixed or args.epochs) + 1):
        enc.train()
        for b in batches(split["fit"], args.batch, True, gen):
            z = enc(x[b].to(DEV))
            loss = F.cross_entropy(scale * task(z), y[b].to(DEV), label_smoothing=smooth)
            if aux is not None and method == "recon":
                c_loss = (1 - F.cosine_similarity(aux(z), x[b].to(DEV), dim=-1)).mean()
            elif aux is not None:
                tgt = dense(M, b).to(DEV)
                c_loss = concept_loss(method, aux(z), tgt)
            if aux is not None:
                loss = loss + lam * c_loss
            if rec is not None:
                loss = loss + recon_lam * (1 - F.cosine_similarity(rec(z), x[b].to(DEV), dim=-1)).mean()
            if method == "unif":  # контроль без концептов и без входа: равномерность на сфере (Wang & Isola, 2020)
                loss = loss + lam * torch.pdist(z).pow(2).mul(-2).exp().mean().log()
            if method == "vicreg":  # ещё один контроль без концептов и без входа
                loss = loss + lam * vicreg_loss(z)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(enc.parameters(), 5.0)
            opt.step()
        enc.eval()
        with torch.no_grad():
            pv = (10.0 * task(enc(x[split["val"]].to(DEV)))).argmax(1).cpu()
        f1 = macro_f1(y[split["val"]], pv, n_cls)
        hist.append(f1)
        if fixed:  # без ранней остановки: последняя эпоха
            best = (f1, ep, {k: v.detach().clone() for k, v in enc.state_dict().items()},
                    {k: v.detach().clone() for k, v in task.state_dict().items()})
        elif best is None or f1 > best[0] + 1e-4:
            best, stale = (f1, ep, {k: v.detach().clone() for k, v in enc.state_dict().items()},
                           {k: v.detach().clone() for k, v in task.state_dict().items()}), 0
        else:
            stale += 1
            if stale >= args.patience:
                break
    enc.load_state_dict(best[2])
    task.load_state_dict(best[3])
    enc.eval()
    with torch.no_grad():
        zt = enc(x[split["test"]].to(DEV))
        pt = (10.0 * task(zt)).argmax(1).cpu()
        # качество выучивания концептов этапа 1 их же головой (AP на тесте)
        a_m = concept_metrics(aux(zt), dense(M, split["test"]).to(DEV)) \
            if aux is not None and method != "recon" and aux_head is None else None
    yt = y[split["test"]]
    enc.task = task  # классификатор задачи нужен для интервенций (d2_cebab_interventions.py, d1_spurious.py)
    enc.aux = aux  # совместно обученная концептная голова (open_vocab.py читает из неё привязку к названиям)
    out = {"val_macro_f1": best[0], "best_epoch": best[1], "test_macro_f1": macro_f1(yt, pt, n_cls),
           "test_acc": float((pt == yt).float().mean())}
    if a_m is not None:
        out.update(a_ap=a_m["ap"], a_macro_ap=a_m["macro_ap"])
    return enc, out


def stage2(z_all, split, M, args, gen, return_scores=False):
    """Новый линейный декодер концептов на замороженных представлениях; эпоха выбирается по val AP.
    return_scores -- вернуть также оценки декодера на тесте (CPU)."""
    gate = concept_head(z_all.shape[1], M.shape[1]).to(DEV)
    opt = torch.optim.AdamW(gate.parameters(), lr=1e-2, weight_decay=0.01)
    val_t = dense(M, split["val"]).to(DEV)
    z_val = z_all[split["val"]].to(DEV)
    best = None
    for ep in range(1, args.gate_epochs + 1):
        for b in batches(split["fit"], 1024, True, gen):
            loss = bce_concepts(gate(z_all[b].to(DEV)), dense(M, b).to(DEV))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        with torch.no_grad():
            ap = concept_metrics(gate(z_val), val_t)["ap"]
        if best is None or ap > best[0]:
            best = (ap, ep, {k: v.detach().clone() for k, v in gate.state_dict().items()})
    gate.load_state_dict(best[2])
    with torch.no_grad():
        scores = gate(z_all[split["test"]].to(DEV))
        m = concept_metrics(scores, dense(M, split["test"]).to(DEV))
    m["gate_epoch"] = best[1]
    m["val_ap"] = best[0]
    return (m, scores.cpu()) if return_scores else m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--dict", default="ngram")
    ap.add_argument("--k", type=int, default=0, help="размер словаря (top-k по частоте), 0 -- весь")
    ap.add_argument("--emb", default="mqa")
    ap.add_argument("--methods", nargs="+", default=["task", "bce", "rank"])
    ap.add_argument("--lambdas", nargs="+", type=float, default=[1.0])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--random-concepts", action="store_true")
    ap.add_argument("--raw-reference", action="store_true", help="декодер этапа 2 прямо на x")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--gate-epochs", type=int, default=30)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    out = HERE / "results"
    out.mkdir(exist_ok=True)
    for seed in args.seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        gen = torch.Generator().manual_seed(seed)
        x, y, split, M = load(args.dataset, args.dict, args.k, args.emb, args.random_concepts, seed)
        common = {"dataset": args.dataset, "dict": args.dict, "k": M.shape[1], "emb": args.emb, "seed": seed,
                  "random_concepts": args.random_concepts, "tag": args.tag}
        if args.raw_reference:
            t0 = time.time()
            m = stage2(F.normalize(x, dim=-1), split, M, args, gen)
            row = {**common, "method": "raw_x", "lambda": None, "concepts": m, "seconds": time.time() - t0}
            print(json.dumps(row), flush=True)
            with open(out / f"{args.dataset}.jsonl", "a") as fh:
                fh.write(json.dumps(row) + "\n")
        for method in args.methods:
            for lam in ([0.0] if method == "task" else args.lambdas):
                t0 = time.time()
                enc, task_m = stage1(x, y, split, M, method, lam, args, gen)
                with torch.no_grad():
                    z_all = torch.cat([enc(x[i:i + 4096].to(DEV)).cpu() for i in range(0, len(x), 4096)])
                m = stage2(z_all, split, M, args, gen)
                row = {**common, "method": method, "lambda": lam, "task": task_m, "concepts": m,
                       "seconds": time.time() - t0}
                print(json.dumps(row), flush=True)
                with open(out / f"{args.dataset}.jsonl", "a") as fh:
                    fh.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
