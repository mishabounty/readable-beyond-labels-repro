#!/usr/bin/env python
"""Сводка: на что раскладывается выигрыш концептной supervision в читаемости отложенных концептов.

Для каждого набора прогонов (метрика читаемости -- AP свежего декодера на отложенных концептах):
  task       -- только задача;
  без смысла -- лучшая (по среднему над seed) конфигурация из recon / unif / rand / cbm_rand / cem_rand;
  концепты   -- лучшая конфигурация из bce / rank / cbm / cem;
  защита     = без смысла - task;  смысл = концепты - без смысла (среднее и разброс по seed, попарно).
Число классов задачи и эффективный ранг z у task показывают, насколько задача схлопывает представление.
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CORA = HERE.parents[0] / "EmbeddedCodebookInterpretability/CodebookReviewChecks/outputs/cora_collapse.jsonl"
SETS = [  # (имя, файл, метрика, число классов, что на этапе 1 -> что читается)
    ("CEBaB", "collapse_cebab.jsonl", "ap_full", 5, "n-граммы -> аспекты людей"),
    ("GoEmotions", "collapse_goemotions.jsonl", "ap_full", 4, "n-граммы -> эмоции людей"),
    ("GoEmotions", "collapse_goemotions_humanAB.jsonl", "ap_full", 4, "13 эмоций -> 14 других"),
    ("CUB, 200 видов", "collapse_cub_AB.jsonl", "ap_full", 200, "атрибуты A -> атрибуты B"),
    ("CUB, 4 надкласса", "collapse_cub4_AB.jsonl", "ap_full", 4, "атрибуты A -> атрибуты B"),
    ("CUB, 200 видов", "collapse_cub_cls.jsonl", "ap_full", 200, "атрибуты вида A -> B (Koh)"),
    ("CUB, 4 надкласса", "collapse_cub4_cls.jsonl", "ap_full", 4, "атрибуты вида A -> B (Koh)"),
    ("CUB, 4 надкл., дообуч.", "finetune_cub4.jsonl", "ap_full", 4, "атрибуты A -> B, обучаемый layer4"),
    ("GoEmotions, дообуч.", "finetune_goemotions.jsonl", "ap_full", 4, "13 эмоций -> 14, обучаемые 2 слоя"),
    ("Cora (GFT)", str(CORA), "ap_B_center", 7, "n-граммы A -> n-граммы B"),
]
FREE = {"recon", "unif", "rand", "cbm_rand", "cem_rand", "rank_rand"}
CONCEPT = {"bce", "rank", "cbm", "cem"}


def main():
    for metric_override in (None, "ap_200", "auc_within"):
        table(metric_override)


def table(metric_override):
    print(f"\n=== метрика: {metric_override or 'AP (все метки)'}")
    print(f"{'набор':24s} {'обучение -> чтение':32s} {'кл.':>4s} {'ранг':>5s} {'вход':>6s} {'task':>6s} "
          f"{'без смысла':>16s} {'концепты':>16s} {'защита':>7s} {'смысл (seed)':>16s}")
    for name, fname, metric, n_cls, what in SETS:
        path = Path(fname) if fname.startswith("/") else HERE / "results" / fname
        if not path.exists():
            continue
        rows = [json.loads(l) for l in open(path)]
        if metric_override:
            if not all(metric_override in r for r in rows):
                continue
            metric = metric_override
        raw = [r[metric] for r in rows if r["method"] == "raw_x"]
        by = defaultdict(dict)  # (метод, lambda) -> {seed: значение}
        for r in rows:
            if r["method"] != "raw_x":
                by[(r["method"], r.get("lambda", 1.0))][r["seed"]] = r[metric]
        seeds = sorted(set.intersection(*[set(v) for v in by.values()]))
        mean = {k: np.mean([v[s] for s in seeds]) for k, v in by.items()}
        if not any(k[0] in FREE for k in by) or not any(k[0] in CONCEPT for k in by):
            print(f"{name:24s} {what:32s} -- ещё не все методы посчитаны")
            continue
        task_key = next(k for k in by if k[0] == "task")
        free = max((k for k in by if k[0] in FREE), key=mean.get)
        conc = max((k for k in by if k[0] in CONCEPT), key=mean.get)
        diff = [by[conc][s] - by[free][s] for s in seeds]
        erank = np.mean([r["effective_rank"] for r in rows if r["method"] == "task"])
        print(f"{name:24s} {what:32s} {n_cls:4d} {erank:5.0f} {raw[0] if raw else float('nan'):6.3f} "
              f"{mean[task_key]:6.3f} {mean[free]:6.3f} {free[0] + ':' + format(free[1], 'g'):>9s} "
              f"{mean[conc]:6.3f} {conc[0] + ':' + format(conc[1], 'g'):>9s} {mean[free] - mean[task_key]:+7.3f} "
              f"{np.mean(diff):+6.3f}±{np.std(diff):.3f} (n={len(seeds)})")


if __name__ == "__main__":
    main()
