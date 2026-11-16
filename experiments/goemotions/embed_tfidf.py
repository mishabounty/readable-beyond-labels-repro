#!/usr/bin/env python
"""Признаки без предобучения: TF-IDF (слова и биграммы, 4096 самых частых по fit, sublinear tf, L2-норма).
Словарь и idf считаются только по fit. Энкодер поверх них обучается целиком с нуля -- постановка, где смысл не может
прийти из предобученного энкодера. Результат: data/<датасет>/emb_tfidf.npy (float16)."""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("datasets", nargs="+")
ap.add_argument("--features", type=int, default=4096)
args = ap.parse_args()
for name in args.datasets:
    d = HERE / "data" / name
    texts = json.loads((d / "texts.json").read_text())
    fit = np.load(d / "split.npz")["fit"]
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_features=args.features, sublinear_tf=True)
    vec.fit([texts[i] for i in fit])
    X = vec.transform(texts).astype(np.float32)
    empty = int((X.getnnz(1) == 0).sum())
    np.save(d / "emb_tfidf.npy", X.toarray().astype(np.float16))
    print(name, X.shape, "пустых строк:", empty, flush=True)
