#!/usr/bin/env python
"""Чтение новых концептов по названию: декодер, привязанный к тексту названия концепта.

Голова TiedHead: логит концепта c для текста v равен s * <z_v, normalize(W e_c)> + b_c, где e_c -- эмбеддинг
названия концепта той же замороженной текстовой моделью (n-грамма -- её текст), W -- общая матрица 768 x 768.
Такая голова переносится на любой концепт, у которого есть название, без единой метки.

Этап 1 (словарь ngram top-1000): task / recon / bce / bce+recon / tied / tied+recon (+recon -- восстановление x
с весом 10 вдобавок к концептной потере).
Этап 2 на замороженном z, концепты людей (aspect для CEBaB, emotion для GoEmotions) на этапе 1 не участвуют:
  * zs_post   -- без меток людей: на z заново обучается TiedHead на n-граммах (эпоха по val macro-AP n-грамм),
                 затем концепты людей читаются по названиям;
  * zs_joint  -- то же, но W берётся из головы, обученной на этапе 1 (только для tied);
  * zs_cos    -- для сырого x: просто косинус x с эмбеддингом названия (нулевой уровень самой текстовой модели);
  * n меток   -- декодер на n текстах (как в transfer_fewshot.py), со случайной инициализацией и с инициализацией
                 направлениями из zs_post (name);
  * full      -- стандартный декодер train.stage2 на всех метках.
Для чтения по названию основная метрика -- macro-AP (не зависит от сдвигов концептов, которых у нового концепта нет).
Результат: results/open_vocab_<датасет>.jsonl.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

import train as T
from transfer_fewshot import Args

HERE = Path(__file__).resolve().parent

# названия концептов людей выбраны до запуска и не подбирались по результатам
NAMES = {
    "aspect": {"food:Positive": "delicious food", "food:Negative": "bad food",
               "service:Positive": "great service", "service:Negative": "terrible service",
               "ambiance:Positive": "nice ambiance", "ambiance:Negative": "unpleasant ambiance",
               "noise:Positive": "quiet restaurant", "noise:Negative": "loud noisy restaurant"},
}


class TiedHead(nn.Module):
    def __init__(self, E):
        super().__init__()
        self.register_buffer("E", E)
        self.W = nn.Linear(E.shape[1], E.shape[1], bias=False)
        nn.init.eye_(self.W.weight)
        self.scale = nn.Parameter(torch.tensor(10.0))
        self.bias = nn.Parameter(torch.full((E.shape[0],), -1.0))

    def directions(self, E=None):
        return F.normalize(self.W(self.E if E is None else E), dim=-1)

    def forward(self, z):
        return self.scale * z @ self.directions().t() + self.bias


def name_embeddings(ds, eval_dict, k):
    """Эмбеддинги названий: n-граммы top-k и концепты людей; кэш в data/<ds>/names_<eval_dict>_<k>.npz."""
    d = HERE / "data" / ds
    cache = d / f"names_{eval_dict}_{k}.npz"
    if not cache.exists():
        from sentence_transformers import SentenceTransformer
        st = SentenceTransformer("sentence-transformers/multi-qa-distilbert-cos-v1", device=str(T.DEV))
        ng = json.loads((d / "concepts_ngram.json").read_text())[:k]
        hu = json.loads((d / f"concepts_{eval_dict}.json").read_text())
        hu = [NAMES.get(eval_dict, {}).get(c, c) for c in hu]
        enc = lambda t: st.encode(t, batch_size=256, normalize_embeddings=True)  # noqa: E731
        np.savez(cache, ngram=enc(ng), human=enc(hu))
        del st
        torch.cuda.empty_cache()
    z = np.load(cache)
    return torch.tensor(z["ngram"]), torch.tensor(z["human"])


@torch.no_grad()
def read_by_name(z, idx, M_eval, U):
    return T.concept_metrics(z[idx].to(T.DEV) @ U.to(T.DEV).t(), T.dense(M_eval, idx).to(T.DEV))


def fit_tied(z_all, split, M_ngram, E_ng, gen, epochs=30):
    head = TiedHead(E_ng).to(T.DEV)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-2, weight_decay=0.01)
    val_t = T.dense(M_ngram, split["val"]).to(T.DEV)
    best = None
    for _ in range(epochs):
        for b in T.batches(split["fit"], 1024, True, gen):
            loss = T.bce_concepts(head(z_all[b].to(T.DEV)), T.dense(M_ngram, b).to(T.DEV))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        with torch.no_grad():
            ap = T.concept_metrics(head(z_all[split["val"]].to(T.DEV)), val_t)["macro_ap"]
        if best is None or ap > best[0]:
            best = (ap, {k: v.detach().clone() for k, v in head.state_dict().items()})
    head.load_state_dict(best[1])
    return head


def fewshot(z_all, split, M, n, seed, init=None, steps=300):
    g = torch.Generator().manual_seed(1000 + seed + n)
    fit = split["fit"][torch.randperm(len(split["fit"]), generator=g)[:n]]
    gate = T.concept_head(z_all.shape[1], M.shape[1]).to(T.DEV)
    if init is not None:
        with torch.no_grad():
            gate.weight.copy_(10.0 * init.to(T.DEV))
    opt = torch.optim.AdamW(gate.parameters(), lr=1e-2, weight_decay=0.01)
    zf, tf = z_all[fit].to(T.DEV), T.dense(M, fit).to(T.DEV)
    for _ in range(steps):
        b = torch.randint(len(fit), (min(len(fit), 1024),), generator=g).to(T.DEV)
        loss = T.bce_concepts(gate(zf[b]), tf[b])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    with torch.no_grad():
        return T.concept_metrics(gate(z_all[split["test"]].to(T.DEV)), T.dense(M, split["test"]).to(T.DEV))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--eval-dict", required=True)
    ap.add_argument("--k", type=int, default=1000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--shots", type=int, nargs="+", default=[50, 200])
    ap.add_argument("--runs", nargs="+",
                    default=["raw_x", "task", "recon", "bce", "bce+recon", "tied", "tied+recon"])
    args = ap.parse_args()
    E_ng, E_hu = name_embeddings(args.dataset, args.eval_dict, args.k)
    for seed in args.seeds:
        _, _, _, M_eval = T.load(args.dataset, args.eval_dict, 0, "mqa", False, seed)
        x, y, split, M_ng = T.load(args.dataset, "ngram", args.k, "mqa", False, seed)
        for run in args.runs:
            torch.manual_seed(seed)
            gen = torch.Generator().manual_seed(seed)
            res, task_m = {}, {}
            if run == "raw_x":
                z_all = F.normalize(x, dim=-1)
                res["zs_cos"] = read_by_name(z_all, split["test"], M_eval, E_hu)
            else:
                base = run.split("+")[0]
                recon_lam = 10.0 if run.endswith("+recon") else 0.0
                method, lam = ("recon", 10.0) if run == "recon" else (("task", 0.0) if run == "task" else ("bce", 1.0))
                aux_head = TiedHead(E_ng) if base == "tied" else None
                enc, task_m = T.stage1(x, y, split, M_ng, method, lam, Args, gen, aux_head=aux_head,
                                       recon_lam=recon_lam)
                with torch.no_grad():
                    z_all = torch.cat([enc(x[i:i + 4096].to(T.DEV)).cpu() for i in range(0, len(x), 4096)])
                    if base == "tied":
                        res["zs_joint"] = read_by_name(z_all, split["test"], M_eval, enc.aux.directions(E_hu.to(T.DEV)))
                del enc
                torch.cuda.empty_cache()
            head = fit_tied(z_all, split, M_ng, E_ng, gen)
            with torch.no_grad():
                U_post = head.directions(E_hu.to(T.DEV)).cpu()
            del head
            res["zs_post"] = read_by_name(z_all, split["test"], M_eval, U_post)
            res["full"] = T.stage2(z_all, split, M_eval, Args, gen)
            for n in args.shots:
                res[f"{n}"] = fewshot(z_all, split, M_eval, n, seed)
                res[f"{n}_name"] = fewshot(z_all, split, M_eval, n, seed, init=U_post)
            row = {"dataset": args.dataset, "eval_dict": args.eval_dict, "run": run, "seed": seed, "task": task_m,
                   "concepts": res}
            with open(HERE / "results" / f"open_vocab_{args.dataset}.jsonl", "a") as fh:
                fh.write(json.dumps(row) + "\n")
            print(json.dumps({"ds": args.dataset, "seed": seed, "run": run,
                              "F1": round(task_m.get("test_macro_f1", float("nan")), 3),
                              **{f"{k}": round(v["macro_ap"], 3) for k, v in res.items()}}), flush=True)


if __name__ == "__main__":
    main()
