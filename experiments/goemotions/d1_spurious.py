#!/usr/bin/env python
"""D1: находится ли и удаляется ли ложная зависимость через декодер концептов (AG News).

В обучающих и валидационных текстах класса Sports в 90% случаев (в остальных классах в 5%) в конец дописывается
фраза « (Zebra Wire)» -- ложный признак класса. Тесты: clean (без фразы) и shifted (фраза у всех НЕ-Sports, у Sports
её нет). Словарь: 1000 n-грамм + концепт «zebra wire» (присутствие фразы).
Для task / bce / rank (lambda = 1, seed 0):
  * поиск: место концепта «zebra wire» среди 1001 концептов по влиянию на логит Sports
           (проекция нормированной строки декодера на направление w_Sports - mean_k w_k), AP декодера по нему;
  * правка без переобучения: из z вычитается проекция на направление концепта в декодере, затем normalize;
           сравнение macro-F1 на clean / shifted до и после; ориентир -- направление разности средних z
           с фразой и без (по fit, требует разметки фразы).
Результат: results/d1_spurious.json.
"""

import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

import train as T
from d2_cebab_interventions import Args, stage2_gate

HERE = Path(__file__).resolve().parent
import argparse

PHRASE = " (Zebra Wire)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="agnews")
    ap.add_argument("--target", type=int, default=1, help="класс, с которым связана ложная фраза")
    ap.add_argument("--p-target", type=float, default=0.9)
    ap.add_argument("--p-other", type=float, default=0.05)
    args = ap.parse_args()
    SPORTS = args.target
    d = HERE / "data" / args.dataset
    texts = json.loads((d / "texts.json").read_text())
    y = torch.tensor(np.load(d / "labels.npy"), dtype=torch.long)
    s = np.load(d / "split.npz")
    split = {k: torch.tensor(s[k]) for k in ("fit", "val", "test")}
    x = torch.tensor(np.load(d / "emb_mqa.npy").astype(np.float32))
    rng = np.random.default_rng(0)
    train_idx = np.concatenate([s["fit"], s["val"]])
    has = np.zeros(len(texts), dtype=bool)
    for i in train_idx:
        has[i] = rng.random() < (args.p_target if y[i] == SPORTS else args.p_other)
    test = s["test"]
    shifted = test[y[test].numpy() != SPORTS]

    st = SentenceTransformer("sentence-transformers/multi-qa-distilbert-cos-v1", device="cuda")
    st.max_seq_length = 256
    enc_txt = lambda idx: torch.tensor(st.encode([texts[i] + PHRASE for i in idx], batch_size=64,  # noqa: E731
                                                 normalize_embeddings=True))
    mod = np.where(has)[0]
    x[mod] = enc_txt(mod)
    x_shift = x.clone()
    x_shift[shifted] = enc_txt(shifted)
    del st

    M = sp.load_npz(d / "concepts_ngram.npz").tocsr()[:, :1000]
    spur_col = sp.csr_matrix(has.astype(np.int8)[:, None])
    M = sp.hstack([M, spur_col]).tocsr()
    S = M.shape[1] - 1  # индекс концепта «zebra wire»
    n_cls = int(y.max()) + 1
    results = {"train_with_phrase": int(has.sum()), "shifted_test": int(len(shifted))}

    def f1(head, z, idx):
        with torch.no_grad():
            pred = (10.0 * head(z.to(T.DEV))).argmax(1).cpu()
        return T.macro_f1(y[idx], pred, n_cls)

    for method, lam in (("task", 0.0), ("bce", 1.0), ("rank", 1.0)):
        torch.manual_seed(0)
        gen = torch.Generator().manual_seed(0)
        enc, task_m = T.stage1(x, y, split, M, method, lam, Args, gen)
        head = enc.task
        with torch.no_grad():
            z_all = torch.cat([enc(x[i:i + 4096].to(T.DEV)).cpu() for i in range(0, len(x), 4096)])
            z_shift = torch.cat([enc(x_shift[i:i + 4096].to(T.DEV)).cpu() for i in range(0, len(x), 4096)])
        gate = stage2_gate(z_all, split, M, gen)
        with torch.no_grad():
            U = F.normalize(gate.weight, dim=-1)
            W = head.weight
            infl = (U @ (W[SPORTS] - W.mean(0))).cpu()
            rank_spur = int((infl > infl[S]).sum()) + 1
            # AP декодера по концепту «zebra wire» на объединении clean + shifted теста
            zt = torch.cat([z_all[test], z_shift[shifted]]).to(T.DEV)
            tt = torch.cat([torch.zeros(len(test)), torch.ones(len(shifted))]).to(T.DEV)
            ap_spur = T.concept_metrics(gate(zt)[:, S:S + 1], tt[:, None])["ap"]
            u = U[S].cpu()
            fit = split["fit"].numpy()
            mu = F.normalize(z_all[fit][has[fit]].mean(0) - z_all[fit][~has[fit]].mean(0), dim=0)

            def edit(z, direction):
                return F.normalize(z - (z @ direction)[:, None] * direction[None], dim=-1)

            row = {"task": task_m, "spurious_rank_by_sports_influence": rank_spur, "spurious_concept_ap": ap_spur,
                   "cos(decoder_dir, mean_diff_dir)": float(u @ mu)}
            for name, dirn in (("before", None), ("edit_decoder_dir", u), ("edit_mean_diff_dir", mu)):
                zc = z_all[test] if dirn is None else edit(z_all[test], dirn)
                zs = z_shift[shifted] if dirn is None else edit(z_shift[shifted], dirn)
                with torch.no_grad():
                    acc_s = float(((10.0 * head(zs.to(T.DEV))).argmax(1).cpu() == y[shifted]).float().mean())
                row[name] = {"clean_f1": f1(head, zc, test), "shifted_acc": acc_s,
                             "shifted_pred_sports": float(((10.0 * head(zs.to(T.DEV))).argmax(1) == SPORTS)
                                                          .float().mean())}
            top = infl.topk(5).indices.tolist()
            names = json.loads((d / "concepts_ngram.json").read_text())[:1000] + ["zebra wire"]
            row["top5_sports_concepts"] = [names[i] for i in top]
        results[method] = row
        print(method, json.dumps(row, ensure_ascii=False), flush=True)
    (HERE / "results" / f"d1_spurious_{args.dataset}.json").write_text(json.dumps(results, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
