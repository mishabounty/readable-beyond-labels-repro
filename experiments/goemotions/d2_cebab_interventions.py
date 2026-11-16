#!/usr/bin/env python
"""D2: соответствуют ли направления концептов в z реальным контрфактам (CEBaB).

Модели этапа 1 (task / bce / rank, словарь aspect, seed 0) и свежий декодер аспектов этапа 2 обучаются как в train.py.
Затем по парам CEBaB «исходный отзыв -> правка одного аспекта (food / service / ambiance / noise) в сторону
Positive или Negative» считается:
  * align  -- косинус между реальным сдвигом z_e - z_o и направлением концепта в декодере
              d = u(аспект:цель) - u(аспект:противоположное) (нормированные строки декодера);
  * detect -- доля пар, где логит декодера для цели правки вырос от исходного отзыва к правке;
  * corr   -- корреляция предсказанного интервенцией эффекта на оценку (сдвиг z_o вдоль d на типичную длину
              реального сдвига) с реальным эффектом правки на предсказание модели; sign -- совпадение знака.
Результат: results/d2_cebab[_<словарь этапа 1>]_seed<seed>_runs.json (--train-dict ngram --k 1000 -- перенос с n-грамм).
--probe newton: декодер аспектов -- логистическая регрессия до сходимости (logreg_newton.py, C для каждого аспекта по
val AP, признаки центрированы и поделены на общий скаляр), направление концепта -- нормированный вектор её весов в z;
файл с суффиксом _newton.
--probe massmean: направление концепта -- разность средних z примеров fit с концептом и без него (mass-mean,
Marks & Tegmark, 2023; от процедуры обучения пробы не зависит), логит -- проекция z на него; суффикс _massmean.
"""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

import logreg_newton as N
import train as T

HERE = Path(__file__).resolve().parent
ASPECTS = ["food", "service", "ambiance", "noise"]
NAMES = [f"{a}:{p}" for a in ASPECTS for p in ("Positive", "Negative")]


class Args:
    epochs, patience, batch, lr, gate_epochs = 40, 6, 256, 1e-3, 30


def stage2_gate(z_all, split, M, gen):
    gate = T.concept_head(z_all.shape[1], M.shape[1]).to(T.DEV)
    opt = torch.optim.AdamW(gate.parameters(), lr=1e-2, weight_decay=0.01)
    val_t = T.dense(M, split["val"]).to(T.DEV)
    best = None
    for _ in range(Args.gate_epochs):
        for b in T.batches(split["fit"], 1024, True, gen):
            loss = T.bce_concepts(gate(z_all[b].to(T.DEV)), T.dense(M, b).to(T.DEV))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        with torch.no_grad():
            ap = T.concept_metrics(gate(z_all[split["val"]].to(T.DEV)), val_t)["ap"]
        if best is None or ap > best[0]:
            best = (ap, {k: v.detach().clone() for k, v in gate.state_dict().items()})
    gate.load_state_dict(best[1])
    return gate


class LinearProbe(torch.nn.Module):
    """Логиты z @ W + b; weight -- строки W^T, как у nn.Linear (направления концептов)."""

    def __init__(self, W, b):
        super().__init__()
        self.weight, self.bias = W.t().contiguous(), b

    def forward(self, z):
        return z @ self.weight.t() + self.bias


def newton_gate(z_all, split, M):
    from probe_eval import CS
    fit, val = split["fit"], split["val"]
    mu = z_all[fit].mean(0)
    sc = (z_all[fit] - mu).pow(2).mean().sqrt()
    Xf = N.add_bias(((z_all[fit] - mu) / sc).to(T.DEV))
    Xv = N.add_bias(((z_all[val] - mu) / sc).to(T.DEV))
    Yf, Yv = T.dense(M, fit).to(T.DEV), T.dense(M, val).to(T.DEV)
    best, Wb, W = torch.full((M.shape[1],), -1.0, device=T.DEV), None, None
    for C in CS:
        W, ok = N.fit(Xf, Yf, C, W0=W)
        ap = N.ap_columns(Xv @ W, Yv).nan_to_num(0.0)
        Wb = W.clone() if Wb is None else torch.where((ap > best)[None], W, Wb)
        best = torch.maximum(best, ap)
    w = Wb[:-1] / sc.to(T.DEV)  # обратно в координаты z
    b = Wb[-1] - mu.to(T.DEV) @ w
    return LinearProbe(w, b)


def massmean_gate(z_all, split, M):
    fit = split["fit"]
    Z, Y = z_all[fit].to(T.DEV), T.dense(M, fit).to(T.DEV)
    pos = (Y.t() @ Z) / Y.sum(0).clamp_min(1)[:, None]
    neg = ((1 - Y).t() @ Z) / (1 - Y).sum(0).clamp_min(1)[:, None]
    w = (pos - neg).t()  # d x m
    b = -(((pos + neg) / 2) * w.t()).sum(1)
    return LinearProbe(w, b)


def pairs_from_test():
    t = load_dataset("CEBaB/CEBaB")["test"]
    orig = {ex["original_id"]: ex["description"] for ex in t if ex["is_original"]}
    pairs = []
    for ex in t:
        if ex["is_original"] or ex["edit_goal"] not in ("Positive", "Negative") or ex["original_id"] not in orig:
            continue
        pairs.append((orig[ex["original_id"]], ex["description"], ex["edit_type"], ex["edit_goal"]))
    return pairs


def expected_rating(task, z):
    p = torch.softmax(10.0 * task(z), dim=-1)
    return (p * torch.arange(1, p.shape[1] + 1, device=p.device)).sum(-1)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("seed", type=int, nargs="?", default=0)
    ap.add_argument("--train-dict", default="aspect", help="словарь этапа 1: aspect или ngram (перенос)")
    ap.add_argument("--k", type=int, default=0)
    ap.add_argument("--random-concepts", action="store_true", help="контроль: случайные концепты той же плотности")
    ap.add_argument("--runs", nargs="+", default=["task:0", "bce:1", "rank:1"], help="метод:lambda")
    ap.add_argument("--probe", default="adam", choices=["adam", "newton", "massmean"])
    args = ap.parse_args()
    seed = args.seed
    pairs = pairs_from_test()
    st = SentenceTransformer("sentence-transformers/multi-qa-distilbert-cos-v1", device="cuda")
    st.max_seq_length = 256
    enc_o = torch.tensor(st.encode([p[0] for p in pairs], batch_size=64, normalize_embeddings=True))
    enc_e = torch.tensor(st.encode([p[1] for p in pairs], batch_size=64, normalize_embeddings=True))
    del st
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    x, y, split, M = T.load("cebab", "aspect", 0, "mqa", False, seed)
    M_train = M if args.train_dict == "aspect" else T.load("cebab", args.train_dict, args.k, "mqa",
                                                                         args.random_concepts, seed)[3]
    results = {"pairs": len(pairs), "train_dict": f"{args.train_dict}{args.k or ''}"}
    for run in args.runs:
        method, lam = run.split(":")[0], float(run.split(":")[1])
        torch.manual_seed(seed)
        enc, task_m = T.stage1(x, y, split, M_train, method, lam, Args, gen)
        head = enc.task
        with torch.no_grad():
            z_all = torch.cat([enc(x[i:i + 4096].to(T.DEV)).cpu() for i in range(0, len(x), 4096)])
            zo, ze = enc(enc_o.to(T.DEV)), enc(enc_e.to(T.DEV))
        gate = {"newton": lambda: newton_gate(z_all, split, M), "massmean": lambda: massmean_gate(z_all, split, M),
                "adam": lambda: stage2_gate(z_all, split, M, gen)}[args.probe]()
        with torch.no_grad():
            U = F.normalize(gate.weight, dim=-1)
            d = torch.stack([U[NAMES.index(f"{a}:{g}")] - U[NAMES.index(f"{a}:{'Negative' if g == 'Positive' else 'Positive'}")]
                             for _, _, a, g in pairs])
            d = F.normalize(d, dim=-1)
            delta = ze - zo
            align = F.cosine_similarity(delta, d, dim=-1)
            tgt = torch.tensor([NAMES.index(f"{a}:{g}") for _, _, a, g in pairs], device=T.DEV)
            lo, le = gate(zo).gather(1, tgt[:, None]).squeeze(1), gate(ze).gather(1, tgt[:, None]).squeeze(1)
            real = expected_rating(head, ze) - expected_rating(head, zo)
            step = delta.norm(dim=-1).median()
            zi = F.normalize(zo + step * d, dim=-1)
            pred = expected_rating(head, zi) - expected_rating(head, zo)
            corr = float(np.corrcoef(pred.cpu().numpy(), real.cpu().numpy())[0, 1])
            sign = float(((pred > 0) == (real > 0)).float().mean())
        results[f"{method}{lam:g}"] = {"task": task_m, "align_mean": float(align.mean()), "detect": float((le > lo).float().mean()),
                           "corr_pred_real": corr, "sign_agree": sign, "real_effect_abs": float(real.abs().mean()),
                           "step": float(step)}
        print(f"{method}{lam:g}", json.dumps(results[f"{method}{lam:g}"]), flush=True)
    tag = "" if args.train_dict == "aspect" else f"_{args.train_dict}{args.k or ''}"
    tag += "_random" if args.random_concepts else ""
    tag += "" if args.probe == "adam" else f"_{args.probe}"
    (HERE / "results" / f"d2_cebab{tag}_seed{seed}_runs.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
