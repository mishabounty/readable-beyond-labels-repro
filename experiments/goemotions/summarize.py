#!/usr/bin/env python
"""Сводка results/*.jsonl: таблицы (среднее +- ст. откл. по seed) и график AP от размера словаря.
Пишет results/summary.md и results/ap_vs_vocab.png."""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
NAMES = {"task": "только задача", "bce": "+ BCE", "rank": "+ ранговая", "raw_x": "декодер на x"}


def load():
    rows = []
    for f in sorted(RES.glob("*.jsonl")):
        rows += [json.loads(line) for line in f.read_text().splitlines() if line.strip()]
    return rows


def fmt(v):
    v = np.array(v, dtype=float)
    return f"{v.mean():.3f}" if len(v) == 1 else f"{v.mean():.3f} ± {v.std(ddof=0):.3f}"


def main():
    rows = load()
    groups = defaultdict(list)
    for r in rows:
        key = (r["dataset"], r["dict"], r["k"], r["random_concepts"], r["method"], r["lambda"])
        groups[key].append(r)
    lines = ["# Сводка: ранговая концептная потеря на текстах", "",
             "Среднее ± ст. откл. по seed. AP, macro-AP, R@P70, P@5 -- декодер этапа 2 на тесте.", ""]
    settings = sorted({k[:4] for k in groups}, key=lambda s: (s[0], s[1], s[3], s[2]))
    for ds, dic, k, rnd in settings:
        lines += [f"## {ds} / {dic} / {k} концептов" + (" / СЛУЧАЙНЫЙ словарь" if rnd else ""), "",
                  "| этап 1 | λ | macro-F1 задачи | AP | macro-AP | R@P70 | P@5 | seeds |",
                  "|---|---|---|---|---|---|---|---|"]
        for (d2, c2, k2, r2, m, lam), rs in sorted(groups.items(), key=lambda kv: (["raw_x", "task", "bce", "rank"].index(kv[0][4]), kv[0][5] or 0)):
            if (d2, c2, k2, r2) != (ds, dic, k, rnd):
                continue
            f1 = fmt([r["task"]["test_macro_f1"] for r in rs]) if m != "raw_x" else "—"
            c = lambda key: fmt([r["concepts"][key] for r in rs])  # noqa: E731
            lines.append(f"| {NAMES[m]} | {lam if lam else '—'} | {f1} | {c('ap')} | {c('macro_ap')} | "
                         f"{c('recall_at_p70')} | {c('p_at5')} | {len(rs)} |")
        lines.append("")
    (RES / "summary.md").write_text("\n".join(lines))
    print("\n".join(lines))

    # график: AP от размера словаря n-грамм (λ = 1), по датасетам
    dss = sorted({k[0] for k in groups if k[1] == "ngram" and not k[3]})
    fig, axes = plt.subplots(1, len(dss), figsize=(4.2 * len(dss), 3.4), squeeze=False)
    for ax, ds in zip(axes[0], dss):
        for m, color in (("raw_x", "#999999"), ("task", "#4c72b0"), ("bce", "#c44e52"), ("rank", "#55a868")):
            pts = []
            for (d2, c2, k2, r2, m2, lam), rs in groups.items():
                if d2 == ds and c2 == "ngram" and not r2 and m2 == m and (lam in (None, 0.0, 1.0)):
                    pts.append((k2, np.mean([r["concepts"]["ap"] for r in rs]), np.std([r["concepts"]["ap"] for r in rs])))
            if pts:
                pts.sort()
                ks, mu, sd = zip(*pts)
                ax.errorbar(ks, mu, yerr=sd, marker="o", color=color, label=NAMES[m], capsize=3,
                            ls="--" if m == "raw_x" else "-")
        ax.set_xscale("log")
        ax.set_title(ds)
        ax.set_xlabel("размер словаря")
        ax.set_ylabel("AP концептов (тест)")
        ax.grid(alpha=0.3)
    axes[0][0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RES / "ap_vs_vocab.png", dpi=150)
    print("saved", RES / "ap_vs_vocab.png")


if __name__ == "__main__":
    main()
