#!/usr/bin/env python
"""Reuters-21578 (разбиение ModApte из корпуса nltk) в формате конвейера: data/reuters/.

Второй набор для режима «обучаемый энкодер + задача, не связанная с концептами»: концепты -- темы новостей,
размеченные людьми (многометочные), задача -- квартиль длины текста в словах (границы по fit).
Темы: не меньше MIN_TRAIN документов в обучающей части и MIN_TEST в тестовой; делятся случайно (seed 0) пополам
на A (обучение концептам) и B (отложенные). Обучающая часть ModApte -> fit / val (15% val, seed 0), тестовая -> test.
Запуск: python -I prepare_reuters.py <распакованный каталог reuters с cats.txt>.
"""

import json
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

HERE = Path(__file__).resolve().parent
MIN_TRAIN, MIN_TEST = 30, 10


def save_concepts(d, name, Y, names):
    sp.save_npz(d / f"concepts_{name}.npz", sp.csr_matrix(Y.astype(np.int8)))
    (d / f"concepts_{name}.json").write_text(json.dumps(names, ensure_ascii=False))


def main():
    root = Path(sys.argv[1])
    cats = {}
    for line in (root / "cats.txt").read_text(encoding="latin-1").splitlines():
        p = line.split()
        cats[p[0]] = p[1:]
    ids = sorted(cats, key=lambda k: (not k.startswith("training"), int(k.split("/")[1])))
    texts = [" ".join((root / k).read_text(encoding="latin-1").split()) for k in ids]
    train = np.array([k.startswith("training") for k in ids])
    count = lambda mask, c: sum(c in cats[k] for k, m in zip(ids, mask) if m)  # noqa: E731
    topics = sorted({c for v in cats.values() for c in v})
    keep = [c for c in topics if count(train, c) >= MIN_TRAIN and count(~train, c) >= MIN_TEST]
    Y = np.array([[c in cats[k] for c in keep] for k in ids])
    rng = np.random.default_rng(0)
    tr = np.where(train)[0]
    val = np.sort(rng.choice(tr, int(0.15 * len(tr)), replace=False))
    fit = np.setdiff1d(tr, val)
    test = np.where(~train)[0]
    n = np.array([len(t.split()) for t in texts])
    q = np.quantile(n[fit], [0.25, 0.5, 0.75])
    labels = np.searchsorted(q, n, side="right")
    perm = rng.permutation(len(keep))
    A, B = np.sort(perm[: len(keep) // 2]), np.sort(perm[len(keep) // 2:])
    d = HERE / "data" / "reuters"
    d.mkdir(parents=True, exist_ok=True)
    (d / "texts.json").write_text(json.dumps(texts, ensure_ascii=False))
    np.save(d / "labels.npy", labels.astype(np.int64))
    np.savez(d / "split.npz", fit=fit, val=val, test=test)
    save_concepts(d, "topic", Y, keep)
    save_concepts(d, "topicA", Y[:, A], [keep[i] for i in A])
    save_concepts(d, "topicB", Y[:, B], [keep[i] for i in B])
    meta = {"source": "nltk reuters (ModApte)", "task": "квартиль длины в словах", "length_quartiles": q.tolist(),
            "n_fit": len(fit), "n_val": len(val), "n_test": len(test),
            "topicA": [keep[i] for i in A], "topicB": [keep[i] for i in B]}
    (d / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    print(json.dumps(meta, ensure_ascii=False))


if __name__ == "__main__":
    main()
