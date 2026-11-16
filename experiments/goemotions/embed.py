#!/usr/bin/env python
"""Замороженные эмбеддинги текстов (sentence-transformers, локальный кэш HF): data/<датасет>/emb_<модель>.npy."""
import argparse
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

HERE = Path(__file__).resolve().parent
MODELS = {"mqa": "sentence-transformers/multi-qa-distilbert-cos-v1", "minilm": "sentence-transformers/all-MiniLM-L6-v2"}

ap = argparse.ArgumentParser()
ap.add_argument("datasets", nargs="+")
ap.add_argument("--model", default="mqa", choices=list(MODELS))
ap.add_argument("--max-len", type=int, default=256)
args = ap.parse_args()
model = SentenceTransformer(MODELS[args.model], device="cuda")
model.max_seq_length = args.max_len
for name in args.datasets:
    d = HERE / "data" / name
    texts = json.loads((d / "texts.json").read_text())
    emb = model.encode(texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True, convert_to_numpy=True)
    np.save(d / f"emb_{args.model}.npy", emb.astype(np.float16))
    print(name, emb.shape, flush=True)
