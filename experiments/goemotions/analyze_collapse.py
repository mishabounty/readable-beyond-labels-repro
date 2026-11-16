#!/usr/bin/env python
"""Анализ collapse_check.py: определяется ли читаемость схлопыванием, независимо от вида вспомогательной потери.

Для каждого датасета и метрики читаемости (ap_full, ap_200):
  * корреляция Спирмена читаемости с эффективным рангом и с долей разброса внутри классов по всем прогонам;
  * кривая «без смысла концептов»: линейная регрессия читаемости на log(эффективного ранга) и долю разброса внутри
    классов только по прогонам task / recon / rand / unif; остаток прогонов с концептами (bce / rank) относительно
    неё -- насколько смысл концептов добавляет сверх защиты от схлопывания; для сравнения -- разброс остатков самих
    прогонов без концептов и размах прироста читаемости (лучший прогон минус task).
Рисунок: results/collapse_vs_readability.png.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

HERE = Path(__file__).resolve().parent
COLORS = {"task": "#444444", "bce": "#d62728", "rank": "#ff7f0e", "cbm": "#8c564b", "cem": "#e377c2",
          "recon": "#1f77b4", "rand": "#9467bd", "cbm_rand": "#bcbd22", "cem_rand": "#17becf", "unif": "#2ca02c",
          "raw_x": "#000000"}
CONCEPT = {"bce", "rank", "cbm", "cem"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", default=["cebab", "goemotions"],
                    help="суффиксы файлов results/collapse_<...>.jsonl, например goemotions_humanAB")
    ap.add_argument("--out", default="collapse_vs_readability.png")
    args = ap.parse_args()
    metrics = ("ap_full", "ap_200", "auc_within")
    fig, axes = plt.subplots(len(args.files), len(metrics), figsize=(5.5 * len(metrics), 4 * len(args.files)),
                             squeeze=False)
    for i, ds in enumerate(args.files):
        rows = [json.loads(l) for l in open(HERE / "results" / f"collapse_{ds}.jsonl")]
        runs = [r for r in rows if r["method"] != "raw_x"]
        raw = [r for r in rows if r["method"] == "raw_x"]
        print(f"=== {ds}: {len(runs)} прогонов")
        for j, metric in enumerate(metrics):
            if not all(metric in r for r in runs):
                axes[i, j].axis("off")
                continue
            y = np.array([r[metric] for r in runs])
            er = np.log(np.array([r["effective_rank"] for r in runs]))
            wc = np.array([r["within_class_share"] for r in runs])
            is_c = np.array([r["method"] in CONCEPT for r in runs])
            X = np.column_stack([np.ones(len(runs)), er, wc])
            coef, *_ = np.linalg.lstsq(X[~is_c], y[~is_c], rcond=None)
            resid = y - X @ coef
            task_y = np.mean([r[metric] for r in runs if r["method"] == "task"])
            print(f"  {metric}: Спирмен с эфф. рангом {spearmanr(er, y)[0]:.2f}, с долей внутри классов "
                  f"{spearmanr(wc, y)[0]:.2f}; R^2 кривой без концептов "
                  f"{1 - resid[~is_c].var() / y[~is_c].var():.2f}")
            by = {m: resid[[r["method"] == m for r in runs]].mean() for m in COLORS if any(r["method"] == m for r in runs)}
            print(f"     остаток прогонов с концептами: среднее {resid[is_c].mean():+.3f}; разброс остатков без концептов "
                  f"{resid[~is_c].std():.3f}; прирост лучшего прогона над task {y.max() - task_y:+.3f}")
            print("     остаток по методам:", {m: round(float(v), 3) for m, v in by.items()})
            for m in COLORS:
                sel = [r for r in runs if r["method"] == m]
                if sel:
                    print(f"     {m:5s} эфф.ранг {np.mean([r['effective_rank'] for r in sel]):6.1f}  "
                          f"внутри классов {np.mean([r['within_class_share'] for r in sel]):.2f}  "
                          f"{metric} {np.mean([r[metric] for r in sel]):.3f} (макс {max(r[metric] for r in sel):.3f})  "
                          f"F1 {np.mean([r['task_f1'] for r in sel]):.3f}")
            if raw:
                print(f"     raw_x эфф.ранг {raw[0]['effective_rank']:.1f}  {metric} {raw[0][metric]:.3f}")
            ax = axes[i, j]
            for m in COLORS:
                sel = [r for r in rows if r["method"] == m]
                if sel:
                    ax.scatter([r["effective_rank"] for r in sel], [r[metric] for r in sel], c=COLORS[m], label=m,
                               s=40 if m != "raw_x" else 90, marker="o" if m != "raw_x" else "*", alpha=0.85)
            ax.set_xscale("log")
            ax.set_xlabel("эффективный ранг z (тест)")
            ax.set_ylabel(f"{metric}, отложенные концепты")
            ax.set_title(f"{ds}, {metric}")
            ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(HERE / "results" / args.out, dpi=150)
    print("saved", HERE / "results" / args.out)


if __name__ == "__main__":
    main()
