#!/usr/bin/env python
"""Обучаемый текстовый энкодер: то же, что cub_finetune.py, для GoEmotions (13 эмоций A -> 14 эмоций B).

Энкодер -- тот же multi-qa-distilbert-cos-v1, что даёт замороженные признаки emb_mqa: нижние 4 слоя заморожены
(их выходы предвычисляются один раз), последние 2 слоя дообучаются, затем среднее по токенам, нормировка и MLP как
в train.py. Для скорости: 12 000 текстов fit, 2 000 val, весь test (5 027), длина до 64 токенов.
Вспомогательные потери и измерения -- как в cub_finetune.py; recon восстанавливает исходный эмбеддинг emb_mqa,
raw_x -- он же (вход до дообучения).
Результат дописывается в results/finetune_goemotions.jsonl.
--task length: задача, не связанная с эмоциями, -- квартиль длины текста в словах (границы по fit); суффикс _length
в имени файла и папки z.
hyp -- контроль без смысла, но зависящий от входа: BCE на «концептах» -- случайных гиперплоскостях исходного
эмбеддинга emb_mqa (порог каждой -- квантиль по fit, доля положительных та же, что у соответствующей эмоции A).
km -- то же с псевдо-концептами k-means (структурированы, зависят от входа, но не человеческие); aonly -- только
концепты A без задачи; l2sp / lpft / wise -- методы против забывания (см. cub_finetune.run).
--dataset reuters: Reuters-21578 (prepare_reuters.py), концепты -- темы topicA / topicB, задача -- только length;
fit и val -- все тексты разбиения (их меньше 12 000 / 2 000), размеры пишутся в z/<набор>/sub_split.npz.
"""

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

import train as T
from collapse_check import collapse_metrics, readability
from cub_finetune import run
from transfer_fewshot import fewshot_probe

HERE = Path(__file__).resolve().parent
DATASET = "goemotions"
D = HERE / "data" / DATASET
MAX_LEN, N_FIT, N_VAL, FROZEN = 64, 12000, 2000, 4
BASE = {"goemotions": "emotion", "reuters": "topic"}  # имя словаря концептов: <base>A / <base>B


def length_labels():
    """Квартиль длины текста в словах (границы квартилей -- по fit)."""
    texts = json.loads((D / "texts.json").read_text())
    n = np.array([len(t.split()) for t in texts])
    fit = np.load(D / "split.npz")["fit"]
    q = np.quantile(n[fit], [0.25, 0.5, 0.75])
    return torch.tensor(np.searchsorted(q, n, side="right"), dtype=torch.long)


def task_labels(task):
    """Метки задачи: length -- квартиль длины; иное, кроме sentiment, -- готовый файл data/<набор>/labels_<task>.npy."""
    return length_labels() if task == "length" else torch.tensor(np.load(D / f"labels_{task}.npy"), dtype=torch.long)


def subset(task="sentiment"):
    x, y, split, _ = T.load(DATASET, f"{BASE[DATASET]}A", 0, "mqa", False, 0)
    if task != "sentiment":
        y = task_labels(task)
    rng = np.random.default_rng(0)
    n_fit, n_val = min(N_FIT, len(split["fit"])), min(N_VAL, len(split["val"]))
    fit = np.sort(rng.choice(split["fit"].numpy(), n_fit, replace=False))
    val = np.sort(rng.choice(split["val"].numpy(), n_val, replace=False))
    test = np.sort(split["test"].numpy())
    sub = np.concatenate([fit, val, test])
    local = {"fit": torch.arange(n_fit), "val": torch.arange(n_fit, n_fit + n_val),
             "test": torch.arange(n_fit + n_val, len(sub))}
    return sub, x[sub], y[sub], local


def hyperplane_concepts(x, M_A, fit, seed):
    """Случайные гиперплоскости входа с теми же долями положительных, что у концептов A (по fit)."""
    import scipy.sparse as sp
    rng = np.random.default_rng(20_000 + seed)
    s = F.normalize(x, dim=-1).numpy() @ rng.standard_normal((x.shape[1], M_A.shape[1]))
    p = np.asarray(M_A[fit.numpy()].mean(0)).ravel()
    thr = np.array([np.quantile(s[fit.numpy(), j], 1 - p[j]) for j in range(len(p))])
    return sp.csr_matrix((s > thr).astype(np.int8))


def kmeans_concepts(x, M_A, fit, seed):
    """Псевдо-концепты со структурой, но без человеческого смысла: k-means (k = число концептов A) на исходном
    эмбеддинге; «концепт» j -- косинус к центру j выше квантиля, доля положительных та же, что у концепта j из A."""
    import scipy.sparse as sp
    from sklearn.cluster import KMeans
    xn, f = F.normalize(x, dim=-1).numpy(), fit.numpy()
    km = KMeans(M_A.shape[1], n_init=3, random_state=seed).fit(xn[f])
    c = km.cluster_centers_ / np.linalg.norm(km.cluster_centers_, axis=1, keepdims=True)
    s = xn @ c.T
    p = np.asarray(M_A[f].mean(0)).ravel()
    thr = np.array([np.quantile(s[f, j], 1 - p[j]) for j in range(len(p))])
    return sp.csr_matrix((s > thr).astype(np.int8))


def load_encoder():
    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer("sentence-transformers/multi-qa-distilbert-cos-v1", device="cpu",
                             model_kwargs={"attn_implementation": "eager"})
    return st.tokenizer, st[0].auto_model.eval()


def frozen_hidden(sub, tok, bert, x_sub):
    """Выходы нижних FROZEN слоёв для текстов sub (memmap) и маски; проверка, что полный проход даёт emb_mqa."""
    tag = "" if MAX_LEN == 64 else f"_len{MAX_LEN}"
    cache = D / f"distilbert_l{FROZEN}_sub{tag}.npy"
    mask_file = D / f"distilbert_l{FROZEN}_sub{tag}_mask.npy"
    if not cache.exists():
        texts = json.loads((D / "texts.json").read_text())
        bert = bert.to(T.DEV)
        hid = np.lib.format.open_memmap(cache, mode="w+", dtype=np.float16, shape=(len(sub), MAX_LEN, 768))
        masks = np.zeros((len(sub), MAX_LEN), dtype=np.int8)
        cos = []
        with torch.no_grad():
            for i in range(0, len(sub), 128):
                enc = tok([texts[j] for j in sub[i:i + 128]], padding="max_length", truncation=True,
                          max_length=MAX_LEN, return_tensors="pt").to(T.DEV)
                h = bert.embeddings(enc["input_ids"])
                for layer in bert.transformer.layer[:FROZEN]:
                    h = layer(x=h, attn_mask=enc["attention_mask"])[-1]
                hid[i:i + len(h)] = h.half().cpu().numpy()
                masks[i:i + len(h)] = enc["attention_mask"].cpu().numpy()
                if i < 512:
                    for layer in bert.transformer.layer[FROZEN:]:
                        h = layer(x=h, attn_mask=enc["attention_mask"])[-1]
                    m = enc["attention_mask"].unsqueeze(-1).float()
                    pooled = F.normalize((h * m).sum(1) / m.sum(1), dim=-1).cpu()
                    cos.append(F.cosine_similarity(pooled, x_sub[i:i + len(h)].float(), dim=-1))
        hid.flush()
        np.save(mask_file, masks)
        print(f"проверка: cos(полный проход, emb_mqa) = {torch.cat(cos).mean():.4f}", flush=True)
        bert.cpu()
    return np.load(cache, mmap_mode="r"), np.load(mask_file)


class TextTop(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.backbone = copy.deepcopy(layers)  # верхние слои DistilBERT (дообучаются)
        self.mlp = T.Encoder(768)

    def forward(self, h, mask):
        for layer in self.backbone:
            h = layer(x=h, attn_mask=mask)[-1]
        m = mask.unsqueeze(-1).float()
        return self.mlp(F.normalize((h * m).sum(1) / m.sum(1), dim=-1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--runs", nargs="+", default=["task:0", "bce:1", "rank:1", "rand:1", "recon:3", "unif:1"])
    ap.add_argument("--out-tag", default="", help="суффикс файла результатов и папки z")
    ap.add_argument("--task", default="sentiment", help="sentiment, length или имя файла labels_<task>.npy (country)")
    ap.add_argument("--concepts", default="", help="другое разбиение эмоций: r2 -> emotion_r2A / emotion_r2B")
    ap.add_argument("--dataset", default="goemotions", choices=list(BASE))
    ap.add_argument("--max-len", type=int, default=64, help="длина в токенах (emb_mqa должен быть посчитан с ней же)")
    ap.add_argument("--save-z", action="store_true",
                    help="сохранить z в results/z/finetune_goemotions/ (строки -- тексты подмножества sub.npy)")
    args = ap.parse_args()
    global DATASET, D, MAX_LEN
    DATASET, D, MAX_LEN = args.dataset, HERE / "data" / args.dataset, args.max_len
    if args.task != "sentiment":
        args.out_tag = f"_{args.task}" + (f"_{args.concepts}" if args.concepts else "") + args.out_tag
    elif args.concepts:
        args.out_tag = f"_{args.concepts}" + args.out_tag
    base = BASE[DATASET]
    dA, dB = (f"{base}_{args.concepts}A", f"{base}_{args.concepts}B") if args.concepts else (f"{base}A", f"{base}B")
    sub, x, y, split = subset(args.task)
    if args.save_z:
        zdir = HERE / "results" / "z" / f"finetune_{DATASET}{args.out_tag}"
        zdir.mkdir(parents=True, exist_ok=True)
        np.save(zdir / "sub.npy", sub)
        np.savez(zdir / "sub_split.npz", n_fit=len(split["fit"]), n_val=len(split["val"]))
    tok, bert = load_encoder()
    hid, masks = frozen_hidden(sub, tok, bert, x)
    top = bert.transformer.layer[FROZEN:]

    def fetch(idx):
        return (torch.from_numpy(np.asarray(hid[idx], dtype=np.float32)).to(T.DEV),
                torch.from_numpy(masks[idx].astype(np.int64)).to(T.DEV))

    M_A = T.load(DATASET, dA, 0, "mqa", False, 0)[3][sub]
    M_B = T.load(DATASET, dB, 0, "mqa", False, 0)[3][sub]
    out = HERE / "results" / f"finetune_{DATASET}{args.out_tag}.jsonl"
    for seed in args.seeds:
        M_rand = T.load(DATASET, dA, 0, "mqa", True, seed)[3][sub]
        M_hyp = hyperplane_concepts(x, M_A, split["fit"], seed)
        M_km = kmeans_concepts(x, M_A, split["fit"], seed) if any(r.startswith("km:") for r in args.runs) else None
        # rowperm (план experiments/PLAN_2026-11-03_E1.md): строки меток A перемешаны между текстами внутри каждой части
        # (fit / val / test) -- доли эмоций, их совместная встречаемость и число эмоций на текст те же, связи с текстом нет
        perm = np.arange(M_A.shape[0])
        rngp = np.random.RandomState(30_000 + seed)
        for part in ("fit", "val", "test"):
            idx = split[part].numpy()
            perm[idx] = idx[rngp.permutation(len(idx))]
        M_rowperm = M_A[perm]
        if seed == args.seeds[0] and not (out.exists() and '"raw_x"' in out.read_text()):  # вход -- один раз
            gen = torch.Generator().manual_seed(0)
            xn = F.normalize(x, dim=-1)
            row = {"dataset": DATASET, "method": "raw_x", "lambda": 0.0, "seed": 0, "task_f1": None,
                   **collapse_metrics(xn[split["test"]], y[split["test"]]), **readability(xn, split, M_B, y, gen),
                   "ap_200": fewshot_probe(xn, split, M_B, 200, 0)["ap"]}
            with open(out, "a") as fh:
                fh.write(json.dumps(row) + "\n")
            print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)
        for r in args.runs:
            method, lam = r.split(":")[0], float(r.split(":")[1])
            torch.manual_seed(seed)
            gen = torch.Generator().manual_seed(seed)
            t0 = time.time()
            M = {"rand": M_rand, "hyp": M_hyp, "km": M_km, "rowperm": M_rowperm}.get(method, M_A)
            z_all, task_m = run(fetch, lambda: TextTop(top), x, y, split, M,
                                "bce" if method in ("hyp", "km", "rowperm") else method,
                                lam, gen, epochs=6, patience=2)
            te = split["test"]
            if args.save_z:
                np.save(zdir / f"{method}_{lam:g}_s{seed}.npy", z_all.numpy().astype(np.float16))
            row = {"dataset": DATASET, "method": method, "lambda": lam, "seed": seed,
                   "task_f1": task_m["test_macro_f1"], "best_epoch": task_m["best_epoch"],
                   **collapse_metrics(z_all[te], y[te]), **readability(z_all, split, M_B, y, gen),
                   "ap_200": fewshot_probe(z_all, split, M_B, 200, seed)["ap"], "seconds": time.time() - t0}
            with open(out, "a") as fh:
                fh.write(json.dumps(row) + "\n")
            print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)


if __name__ == "__main__":
    main()
