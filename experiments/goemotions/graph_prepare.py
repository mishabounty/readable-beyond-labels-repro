#!/usr/bin/env python
"""Графы с текстами узлов (Graph-COM/Text-Attributed-Graphs: cora, citeseer, pubmed, wikics) -> data/graph_<имя>/.

Вход узла x -- эмбеддинг его текста той же моделью, что в текстовых постановках (multi-qa-distilbert-cos-v1,
256 токенов). Концепты -- слова текста узла (нижний регистр, буквы, длина >= 3, не стоп-слова), объединённые по
основе Snowball; берутся NWORDS основ с долей документов 1--30%, затем основы случайно (seed 0) делятся пополам на
A и B -- общих основ у A и B нет. Задачи: topic -- исходные классы узлов; degree -- квартиль степени узла
(естественная структурная величина, не связанная со словами текста). Разбиение узлов своё для всех графов:
60 / 20 / 20 (seed 0).
Запуск: python graph_prepare.py <каталог data_raw/tag> [имена].
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

HERE = Path(__file__).resolve().parent
NWORDS = 600


def stems_of(text, stem, cache):
    out = set()
    for w in re.findall(r"[a-z]{3,}", text.lower()):
        if w in ENGLISH_STOP_WORDS:
            continue
        if w not in cache:
            cache[w] = stem.stem(w)
        out.add(cache[w])
    return out


def main():
    root = Path(sys.argv[1])
    names = sys.argv[2:] or ["cora", "citeseer", "pubmed", "wikics"]
    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer("sentence-transformers/multi-qa-distilbert-cos-v1", device="cuda")
    st.max_seq_length = 256
    stem, cache = SnowballStemmer("english"), {}
    for g in names:
        texts = [str(t) for t in torch.load(root / g / "raw_texts.pt", weights_only=False)]
        data = torch.load(root / g / "processed_data.pt", weights_only=False)
        n = len(texts)
        ei = data.edge_index
        ei = torch.cat([ei, ei.flip(0)], 1).unique(dim=1)  # неориентированный граф без повторов
        ei = ei[:, ei[0] != ei[1]]
        deg = torch.bincount(ei[0], minlength=n).numpy()
        rng = np.random.default_rng(0)
        perm = rng.permutation(n)
        fit, val, test = np.sort(perm[: int(0.6 * n)]), np.sort(perm[int(0.6 * n): int(0.8 * n)]), np.sort(perm[int(0.8 * n):])
        q = np.quantile(deg[fit], [0.25, 0.5, 0.75])
        y_degree = np.searchsorted(q, deg, side="right")
        doc = [stems_of(t, stem, cache) for t in texts]
        df = Counter(s for d in doc for s in d)
        cand = [s for s, c in df.most_common() if 0.01 * n <= c <= 0.3 * n][:NWORDS]
        perm_w = rng.permutation(len(cand))
        A = sorted(cand[i] for i in perm_w[: len(cand) // 2])
        B = sorted(cand[i] for i in perm_w[len(cand) // 2:])

        def matrix(words):
            col = {w: j for j, w in enumerate(words)}
            rows, cols = [], []
            for i, d in enumerate(doc):
                for s in d:
                    if s in col:
                        rows.append(i)
                        cols.append(col[s])
            return sp.csr_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(n, len(words)))

        x = st.encode(texts, batch_size=64, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
        d = HERE / "data" / f"graph_{g}"
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "emb_mqa.npy", x.astype(np.float16))
        np.save(d / "edge_index.npy", ei.numpy().astype(np.int64))
        np.save(d / "labels_topic.npy", data.y.numpy().astype(np.int64))
        np.save(d / "labels_degree.npy", y_degree.astype(np.int64))
        np.savez(d / "split.npz", fit=fit, val=val, test=test)
        for name, words in (("wordA", A), ("wordB", B)):
            sp.save_npz(d / f"concepts_{name}.npz", matrix(words))
            (d / f"concepts_{name}.json").write_text(json.dumps(words))
        meta = {"nodes": n, "edges_undirected": int(ei.shape[1] // 2), "classes": int(data.y.max()) + 1,
                "concepts_A": len(A), "concepts_B": len(B), "degree_quartiles": q.tolist(),
                "density_B": float(matrix(B).mean()), "source": "Graph-COM/Text-Attributed-Graphs"}
        (d / "meta.json").write_text(json.dumps(meta, indent=1))
        print(g, json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
