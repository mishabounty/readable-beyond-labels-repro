#!/usr/bin/env python
"""Наглядный пример схлопывания на GoEmotions (замороженный энкодер, seed 0, 13 эмоций A -> 14 эмоций B).

Обучаются три модели этапа 1: только задача (тональность), ранговая потеря на эмоциях A, равномерность.
Для каждой (и для сырого эмбеддинга x) печатается:
  * средний косинус между случайными парами позитивных тестовых комментариев с РАЗНЫМИ эмоциями B
    (насколько модель «склеивает» разные позитивные эмоции);
  * эффективный ранг z;
  * для нескольких конкретных комментариев -- место их настоящей эмоции B среди 14 по оценке свежего декодера
    и косинус между комментариями.
Числа сохраняются в results/example_collapse.json (для рисунка статьи).
"""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import train as T
from collapse_check import collapse_metrics
from transfer_fewshot import Args

HERE = Path(__file__).resolve().parent
KEYS = {"gratitude": ("thank",), "amusement": ("lol", "haha", "hilarious"), "love": ("love",),
        "admiration": ("amazing", "beautiful", "awesome")}


def main():
    texts = json.loads((HERE / "data/goemotions/texts.json").read_text())
    namesB = json.loads((HERE / "data/goemotions/concepts_emotionB.json").read_text())
    x, y, split, M_A = T.load("goemotions", "emotionA", 0, "mqa", False, 0)
    M_B = T.load("goemotions", "emotionB", 0, "mqa", False, 0)[3]
    te = split["test"].numpy()
    B = M_B.toarray()
    # позитивные тестовые комментарии ровно с одной эмоцией B
    pos = [i for i in te if y[i] == 0 and B[i].sum() == 1]
    picked = {}
    for emo, keys in KEYS.items():
        j = namesB.index(emo)
        cand = [i for i in pos if B[i, j] and any(k in texts[i].lower() for k in keys) and len(texts[i].split()) <= 12]
        picked[emo] = cand[0]
    reps = {"x (вход)": F.normalize(x, dim=-1)}
    for name, method, lam in (("только задача", "task", 0.0), ("ранговая на эмоциях A", "rank", 1.0),
                              ("равномерность", "unif", 1.0)):
        torch.manual_seed(0)
        enc, _ = T.stage1(x, y, split, M_A, method, lam, Args, torch.Generator().manual_seed(0))
        with torch.no_grad():
            reps[name] = torch.cat([enc(x[i:i + 4096].to(T.DEV)).cpu() for i in range(0, len(x), 4096)])
    saved = {"comments": {emo: texts[i] for emo, i in picked.items()}, "models": {}}
    print("комментарии:")
    for emo, i in picked.items():
        print(f"  [{emo}] {texts[i]}")
    rng = np.random.default_rng(0)
    pairs = [(a, b) for a, b in (rng.choice(pos, 2, replace=False) for _ in range(3000)) if (B[a] != B[b]).any()]
    for name, z in reps.items():
        zs = F.normalize(z, dim=-1)
        cos_pairs = np.mean([float(zs[a] @ zs[b]) for a, b in pairs])
        er = collapse_metrics(zs[te], y[te])["effective_rank"]
        m, scores = T.stage2(zs, split, M_B, Args, torch.Generator().manual_seed(0), return_scores=True)
        pos_in_test = {i: k for k, i in enumerate(te)}
        ranks = {emo: int((scores[pos_in_test[i]] > scores[pos_in_test[i], namesB.index(emo)]).sum()) + 1
                 for emo, i in picked.items()}
        ids = list(picked.values())
        c = zs[ids] @ zs[ids].t()
        print(f"\n== {name}: эфф. ранг {er:.0f}, AP эмоций B {m['ap']:.3f}, "
              f"средний косинус разных позитивных комментариев {cos_pairs:.2f}")
        print("   место настоящей эмоции среди 14:", ranks)
        print("   косинус благодарность--веселье {:.2f}, благодарность--любовь {:.2f}, веселье--восхищение {:.2f}".format(
            float(c[0, 1]), float(c[0, 2]), float(c[1, 3])))
        saved["models"][name] = {"effective_rank": er, "ap_B": m["ap"], "mean_cos_positive_pairs": cos_pairs,
                                 "cos_matrix": c.tolist(), "true_emotion_rank": ranks}
    (HERE / "results" / "example_collapse.json").write_text(json.dumps(saved, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
