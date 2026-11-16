#!/usr/bin/env python
"""Подготовка текстовых датасетов для проверки ранговой концептной потери без графов.

Для каждого датасета сохраняются тексты, метки задачи, разбиение fit / val / test и словари концептов:
  * ngram  -- автоматический словарь фраз из корпуса (те же правила, что для Cora: 1--3-граммы, частота, связность
              фразы, объединение словоформ по основе); строится без меток и без разбиения, хранится top-5000 по
              документной частоте, меньшие словари -- его префиксы;
  * emotion (GoEmotions) -- 27 эмоций, размеченных людьми (внешние концепты, не слова текста);
  * aspect  (CEBaB)      -- аспекты отзыва (еда, сервис, атмосфера, шум) x {positive, negative}, размечены людьми.

Результат: data/<датасет>/{meta.json, texts.json, labels.npy, split.npz, concepts_<словарь>.npz}.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
STOP = set(ENGLISH_STOP_WORDS)
STEM = SnowballStemmer("english")
TOKEN_RE = re.compile(r"[a-z][a-z0-9]*(?:'[a-z0-9]+)*")
TOP_K = 5000


# ------------------------------------------------------------------ словарь n-грамм
def normalize(text: str) -> str:
    return " ".join(TOKEN_RE.findall(text.lower().replace("-", " ")))


def valid_phrase(phrase: str) -> bool:
    words = phrase.split()
    if words[0] in STOP or words[-1] in STOP:
        return False
    if len(words) == 1:
        return len(words[0]) >= 3
    return sum(w not in STOP for w in words) >= 2 and sum(w in STOP for w in words) <= 1 and \
        all(words[i] != words[i + 1] for i in range(len(words) - 1))


def build_ngram_dictionary(texts, min_df=5, max_df=0.5, min_cohesion=0.08, top_k=TOP_K):
    """Возвращает (названия концептов, разреженная бинарная матрица присутствия [документы x концепты])."""
    clean = [normalize(t) for t in texts]
    vec = CountVectorizer(ngram_range=(1, 3), min_df=min_df, max_df=max_df, binary=True,
                          token_pattern=r"(?u)\b[a-z][a-z0-9']+\b", lowercase=False, dtype=np.int8)
    X = vec.fit_transform(clean).tocsc()
    names = vec.get_feature_names_out()
    df = np.asarray(X.sum(axis=0)).ravel()
    vocab = vec.vocabulary_
    keep = []
    for j, phrase in enumerate(names):
        if not valid_phrase(phrase):
            continue
        words = phrase.split()
        if len(words) > 1:
            content = [vocab[w] for w in words if w not in STOP and w in vocab]
            if not content or df[j] / min(df[c] for c in content) < min_cohesion:
                continue
        keep.append(j)
    # объединение словоформ по основе: network / networks -> один концепт
    groups = defaultdict(list)
    for j in keep:
        groups[" ".join(STEM.stem(w) for w in names[j].split())].append(j)
    cols, labels = [], []
    for members in groups.values():
        rep = max(members, key=lambda j: (df[j], -len(names[j])))
        col = X[:, members].max(axis=1) if len(members) > 1 else X[:, [members[0]]]
        cols.append(sp.csc_matrix(col))
        labels.append(str(names[rep]))
    M = sp.hstack(cols).tocsc().astype(np.int8)
    M.data[:] = 1
    order = np.argsort(-np.asarray(M.sum(axis=0)).ravel(), kind="stable")[:top_k]
    return [labels[i] for i in order], M[:, order].tocsr()


# ------------------------------------------------------------------ датасеты
def stratified_val(labels, train_idx, frac, seed):
    rng = np.random.default_rng(seed)
    val = []
    for c in np.unique(labels[train_idx]):
        idx = train_idx[labels[train_idx] == c]
        val.extend(rng.choice(idx, size=max(1, int(round(frac * len(idx)))), replace=False))
    val = np.array(sorted(val))
    return np.setdiff1d(train_idx, val), val


def load_agnews(seed):
    from datasets import load_dataset
    ds = load_dataset("ag_news")
    rng = np.random.default_rng(seed)
    tr = ds["train"].select(rng.choice(len(ds["train"]), 20000, replace=False))
    te = ds["test"]
    texts = list(tr["text"]) + list(te["text"])
    labels = np.array(list(tr["label"]) + list(te["label"]))
    return texts, labels, ds["train"].features["label"].names, np.arange(len(tr)), np.arange(len(tr), len(texts)), {}


def load_20ng(seed):
    from sklearn.datasets import fetch_20newsgroups
    parts = {s: fetch_20newsgroups(subset=s, remove=("headers", "footers", "quotes")) for s in ("train", "test")}
    texts, labels, split = [], [], []
    for s, b in parts.items():
        for t, y in zip(b.data, b.target):
            if len(t.strip()) >= 30:          # пустые после удаления заголовков сообщения выбрасываются
                texts.append(t.strip()[:4000])
                labels.append(int(y))
                split.append(s)
    split = np.array(split)
    return texts, np.array(labels), list(parts["train"].target_names), np.where(split == "train")[0], \
        np.where(split == "test")[0], {}


GOEMO_GROUPS = {  # отображение эмоций в группы тональности из репозитория авторов GoEmotions
    "positive": ["admiration", "amusement", "approval", "caring", "desire", "excitement", "gratitude", "joy", "love",
                 "optimism", "pride", "relief"],
    "negative": ["anger", "annoyance", "disappointment", "disapproval", "disgust", "embarrassment", "fear", "grief",
                 "nervousness", "remorse", "sadness"],
    "ambiguous": ["confusion", "curiosity", "realization", "surprise"],
    "neutral": ["neutral"],
}


def load_goemotions(seed):
    from datasets import load_dataset
    ds = load_dataset("google-research-datasets/go_emotions", "simplified")
    names = ds["train"].features["labels"].feature.names
    group_of = {e: g for g, es in GOEMO_GROUPS.items() for e in es}
    groups = list(GOEMO_GROUPS)
    emotions = [e for e in names if e != "neutral"]
    texts, labels, emo_rows, split = [], [], [], []
    for s in ("train", "validation", "test"):
        for t, ls in zip(ds[s]["text"], ds[s]["labels"]):
            gs = {group_of[names[l]] for l in ls}
            if len(gs) != 1:                   # смешанная тональность -- пример не используется
                continue
            texts.append(t)
            labels.append(groups.index(gs.pop()))
            emo_rows.append([emotions.index(names[l]) for l in ls if names[l] != "neutral"])
            split.append(s)
    split = np.array(split)
    rows = [i for i, r in enumerate(emo_rows) for _ in r]
    cols = [c for r in emo_rows for c in r]
    E = sp.csr_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(len(texts), len(emotions)))
    fit = np.where(split == "train")[0]
    return texts, np.array(labels), groups, fit, np.where(split == "test")[0], \
        {"emotion": (emotions, E)}, np.where(split == "validation")[0]


def load_cebab(seed):
    from datasets import load_dataset
    ds = load_dataset("CEBaB/CEBaB")
    aspects = ["food", "service", "ambiance", "noise"]
    names = [f"{a}:{p}" for a in aspects for p in ("Positive", "Negative")]
    texts, labels, rows, split = [], [], [], []
    for s in ("train_inclusive", "validation", "test"):
        for ex in ds[s]:
            y = ex["review_majority"]
            if y in ("", "no majority", None):
                continue
            row = [names.index(f"{a}:{ex[f'{a}_aspect_majority']}") for a in aspects
                   if ex[f"{a}_aspect_majority"] in ("Positive", "Negative")]
            texts.append(ex["description"])
            labels.append(int(y) - 1)
            rows.append(row)
            split.append(s)
    split = np.array(split)
    r = [i for i, rr in enumerate(rows) for _ in rr]
    c = [x for rr in rows for x in rr]
    A = sp.csr_matrix((np.ones(len(r), dtype=np.int8), (r, c)), shape=(len(texts), len(names)))
    return texts, np.array(labels), ["1", "2", "3", "4", "5"], np.where(split == "train_inclusive")[0], \
        np.where(split == "test")[0], {"aspect": (names, A)}, np.where(split == "validation")[0]


LOADERS = {"agnews": load_agnews, "20ng": load_20ng, "goemotions": load_goemotions, "cebab": load_cebab}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="*", default=list(LOADERS))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    for name in args.datasets:
        out = DATA / name
        out.mkdir(parents=True, exist_ok=True)
        res = LOADERS[name](args.seed)
        texts, labels, label_names, train, test, extra = res[:6]
        if len(res) == 7:
            fit, val = train, res[6]
        else:
            fit, val = stratified_val(labels, train, 0.15, args.seed)
        concept_names, ngram = build_ngram_dictionary(texts)
        concepts = {"ngram": (concept_names, ngram), **extra}
        for cname, (cnames, M) in concepts.items():
            sp.save_npz(out / f"concepts_{cname}.npz", M.tocsr())
            (out / f"concepts_{cname}.json").write_text(json.dumps(cnames, ensure_ascii=False))
        (out / "texts.json").write_text(json.dumps(texts, ensure_ascii=False))
        np.save(out / "labels.npy", labels)
        np.savez(out / "split.npz", fit=fit, val=val, test=test)
        meta = {"name": name, "n": len(texts), "fit": len(fit), "val": len(val), "test": len(test),
                "classes": label_names,
                "concepts": {c: {"size": M.shape[1], "mean_per_text": float(M.sum() / M.shape[0])}
                             for c, (_, M) in concepts.items()}}
        (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
        print(json.dumps(meta, ensure_ascii=False))


if __name__ == "__main__":
    main()
