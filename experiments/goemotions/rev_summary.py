#!/usr/bin/env python
"""Сводка доработки: проба до сходимости (results/probe_*.jsonl) рядом со старой пробой (results/collapse_*.jsonl),
разложение «форма потери / смысл», парный бутстрап по тестовым примерам (results/scores/), доза-ответ.

Выход: results/rev_summary.md (+ печать). Запуск в любой момент: берёт то, что уже посчитано.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

import train as T
from probe_eval import spec

HERE = Path(__file__).resolve().parent
R = HERE / "results"
METRICS = ["macro_ap", "ap", "auc_within", "macro_ap_200", "ap_200"]
ORDER = ["input", "task", "bce", "bce_bal", "bce_plain", "rank", "cbm", "cem", "rand", "rand_bal", "cbm_rand",
         "cem_rand", "recon", "unif", "vicreg"]
MAIN = {"cebab_v2": "CEBaB, n-grams -> aspects", "goemotions_v2": "GoEmotions, n-grams -> emotions",
        "goemotions_humanAB_v2": "GoEmotions, 13 -> 14 emotions", "cub_AB_v2": "CUB, 200 species",
        "cub4_AB_v2": "CUB, 4 random superclasses", "cub4sem_AB_v2": "CUB, 4 attribute superclasses",
        "finetune_cub4_v2": "CUB-4, trainable ResNet-18 layer4",
        "finetune_goemotions_v2": "GoEmotions 13 -> 14, trainable DistilBERT top-2"}
FIXED = {"recon": 3.0}  # основной вес; остальные -- 1


def load_probe(name):
    f = R / f"probe_{name}.jsonl"
    return [json.loads(l) for l in open(f)] if f.exists() else []


def main_rows(rows, white=False):
    """Строки основного веса: method -> список по seed."""
    out = defaultdict(list)
    for r in rows:
        if r["white"] != white:
            continue
        lam = FIXED.get(r["method"], 1.0) if r["method"] not in ("input", "task") else 0.0
        if r["lambda"] == lam:
            out[r["method"]].append(r)
    return out


def ms(vals):
    vals = [v for v in vals if v is not None and not np.isnan(v)]
    if not vals:
        return "--"
    return f"{np.mean(vals):.3f}" + (f"±{np.std(vals):.3f}" if len(vals) > 1 else "")


# ------------------------------------------------------------------ парный бутстрап
_cache = {}


def test_data(name):
    if name not in _cache:
        ds, ev, emb = spec(name)
        x, y, split, M = T.load(ds, ev, 0, emb, False, 0)
        zdir = R / "z" / name
        if (zdir / "sub.npy").exists():
            sub = np.load(zdir / "sub.npy")
            y, M = y[sub], M[sub]
            te = torch.arange(14000, len(sub))
        else:
            te = split["test"]
        _cache[name] = (torch.tensor(M[te.numpy()].toarray(), dtype=torch.float32, device=T.DEV), y[te].to(T.DEV))
    return _cache[name]


def macro_ap_batch(S, Y, idx):
    """macro-AP для каждой бутстрап-выборки: S, Y -- тест x концепты (GPU), idx -- B x n индексы."""
    out = []
    for b in range(len(idx)):
        Sb, Yb = S[idx[b]], Y[idx[b]]
        cols = Yb.sum(0) >= 5
        o = Sb[:, cols].argsort(0, descending=True)
        Ys = Yb[:, cols].gather(0, o)
        cp = Ys.cumsum(0) / torch.arange(1, len(Ys) + 1, device=Ys.device)[:, None]
        out.append(((cp * Ys).sum(0) / Ys.sum(0)).mean())
    return torch.stack(out)


def bootstrap(name, a, b, B=1000, seed=0):
    """Разность macro-AP (a - b), усреднённая по seed; 95% интервал по тестовым примерам (общие для всех seed)."""
    Y, _ = test_data(name)
    sdir = R / "scores" / name

    def files(spec_):
        m, lam = spec_
        if m == "input":
            return sorted(sdir.glob("input_0_s0.npy"))
        return sorted(sdir.glob(f"{m}_{lam:g}_s*.npy"))

    fa, fb = files(a), files(b)
    if not fa or not fb:
        return None
    g = torch.Generator(device="cpu").manual_seed(seed)
    idx = torch.randint(len(Y), (B, len(Y)), generator=g).to(T.DEV)

    def mean_over(files_):
        acc = 0
        for f in files_:
            S = torch.tensor(np.load(f).astype(np.float32), device=T.DEV)
            acc = acc + macro_ap_batch(S, Y, idx)
        return acc / len(files_)

    full = torch.arange(len(Y), device=T.DEV)[None]
    d = (mean_over(fa) - mean_over(fb)).cpu().numpy()
    point = float(np.mean([macro_ap_batch(torch.tensor(np.load(f).astype(np.float32), device=T.DEV), Y, full).item()
                           for f in fa]) - np.mean(
        [macro_ap_batch(torch.tensor(np.load(f).astype(np.float32), device=T.DEV), Y, full).item() for f in fb]))
    return point, float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), len(fa), len(fb)


# ------------------------------------------------------------------ старая проба
def old_rows(name):
    """Строки старой пробы (collapse_*.jsonl / finetune_*.jsonl) для того же набора z."""
    cand = [R / f"collapse_{name}.jsonl", R / f"{name}.jsonl"]
    for f in cand:
        if f.exists():
            return [json.loads(l) for l in open(f)]
    return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--no-boot", action="store_true")
    args = ap.parse_args()
    lines = ["# Доработка: проба до сходимости, разложение, бутстрап\n"]

    # 1. Основная таблица
    lines.append("## 1. Читаемость отложенных концептов: проба до сходимости (среднее ± s.d. по seed)\n")
    lines.append("macro-AP / micro-AP / AUC внутри класса / macro-AP@200; в скобках -- с отбеливанием (macro-AP); "
                 "eff. rank и task F1 -- из прогонов этапа 1; old AP -- прежняя проба (AdamW 30 эпох).\n")
    for name, title in MAIN.items():
        rows = load_probe(name)
        if not rows:
            continue
        raw, wh = main_rows(rows), main_rows(rows, True)
        old = defaultdict(list)
        for r in old_rows(name):
            key = "input" if r["method"] == "raw_x" else r["method"]
            lam = FIXED.get(key, 1.0) if key not in ("input", "task") else 0.0
            if r["lambda"] == lam:
                old[key].append(r)
        prev = rows[0]["prevalence_macro_ap"]
        lines.append(f"\n### {title} (`{name}`), частотный ориентир macro-AP {prev:.3f}\n")
        lines.append("| объектив | macro-AP | AP | AUC внутри | macro-AP@200 | macro-AP отбел. | old AP | eff. rank | F1 | A-AP |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for m in ORDER:
            if m not in raw:
                continue
            o = old.get(m, [])
            lines.append(f"| {m} | " + " | ".join(ms([r[k] for r in raw[m]]) for k in
                                                  ("macro_ap", "ap", "auc_within", "macro_ap_200")) +
                         f" | {ms([r['macro_ap'] for r in wh.get(m, [])])} | {ms([r.get('ap_full') for r in o])} | "
                         f"{ms([r.get('effective_rank') for r in o])} | {ms([r.get('task_f1') for r in o])} | "
                         f"{ms([r.get('a_ap') for r in o])} |")

    # 2. Разложение
    lines.append("\n## 2. Разложение прироста концептной BCE: форма потери (rand - task) и смысл (bce - rand)\n")
    lines.append("| постановка | метрика | task | rand | bce | доля смысла | лучший без меток (unif/recon) |")
    lines.append("|---|---|---|---|---|---|---|")
    for name, title in MAIN.items():
        raw = main_rows(load_probe(name))
        if not all(k in raw for k in ("task", "rand", "bce")):
            continue
        for k in ("macro_ap", "auc_within"):
            t, r_, b = (np.mean([x[k] for x in raw[m]]) for m in ("task", "rand", "bce"))
            share = (b - r_) / (b - t) if abs(b - t) > 1e-9 else float("nan")
            lf = max((np.mean([x[k] for x in raw[m]]), m) for m in ("unif", "recon") if m in raw)
            lines.append(f"| {title} | {k} | {t:.3f} | {r_:.3f} | {b:.3f} | {share:.0%} | {lf[0]:.3f} ({lf[1]}) |")

    # 3. Бутстрап
    if not args.no_boot:
        lines.append("\n## 3. Парный бутстрап по тестовым примерам: разность macro-AP, 95% интервал "
                     f"({args.boot} выборок; среднее по seed)\n")
        lines.append("| постановка | сравнение | разность | 95% CI |")
        lines.append("|---|---|---|---|")
        pairs = [(("bce", 1.0), ("rand", 1.0)), (("bce", 1.0), ("task", 0.0)), (("bce", 1.0), ("unif", 1.0)),
                 (("rank", 1.0), ("unif", 1.0)), (("unif", 1.0), ("input", 0.0)), (("bce", 1.0), ("input", 0.0)),
                 (("recon", 3.0), ("input", 0.0))]
        for name, title in MAIN.items():
            for a, b in pairs:
                try:
                    res = bootstrap(name, a, b, B=args.boot)
                except FileNotFoundError:
                    res = None
                if res is None:
                    continue
                p, lo, hi, na, nb = res
                star = " *" if lo > 0 or hi < 0 else ""
                lines.append(f"| {title} | {a[0]} - {b[0]} | {p:+.3f}{star} | [{lo:+.3f}, {hi:+.3f}] |")
            _cache.clear()
            torch.cuda.empty_cache()

    text = "\n".join(lines) + "\n"
    (R / "rev_summary.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
