#!/usr/bin/env python
"""Сводка DreamSim с пустышками (план experiments/PLAN_2026-11-10_dreamsim.md). Читает results/eval_*.json.
Разницы:
- human − shuf (парно по сиду k: тот же сид обучения), human − anti (парно по k = 0..2);
- human − base, shuf − base, anti − base (base и authors — без разброса).
95% t-интервалы и бутстрэп по сидам. Итог — stdout и results/summary.json."""

import glob
import json
import os

import numpy as np
from scipy import stats

R = os.path.expanduser("~/dreamsim_repro/results")
ev = {os.path.basename(f)[5:-5]: json.load(open(f)) for f in glob.glob(os.path.join(R, "eval_*.json"))}
METRICS = [("nights_test_2afc", lambda r: 100 * r["nights_test_2afc"]), ("tid2013 srocc", lambda r: r["tid2013"]["srocc"]),
           ("kadid10k srocc", lambda r: r["kadid10k"]["srocc"]), ("tid2013 plcc", lambda r: r["tid2013"]["plcc"]),
           ("kadid10k plcc", lambda r: r["kadid10k"]["plcc"])]


def ci(d):
    d = np.asarray(d, float)
    if len(d) < 2:
        return [float(d.mean()) if len(d) else float("nan"), float("nan"), float("nan")]
    h = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    return [float(d.mean()), float(d.mean() - h), float(d.mean() + h)]


out = {"levels": {}, "diffs": {}}
groups = {g: sorted(k for k in ev if k.startswith(g + "_s")) for g in ("human", "shuf", "anti", "rp")}
print("модели:", {g: len(v) for g, v in groups.items()}, "base" in ev, "authors" in ev)
for mname, f in METRICS:
    lv = {g: [f(ev[k]) for k in ks] for g, ks in groups.items()}
    for single in ("base", "authors"):
        if single in ev:
            lv[single] = [f(ev[single])]
    out["levels"][mname] = {g: ci(v) for g, v in lv.items()}
    print(f"\n{mname}: " + "; ".join(f"{g} {np.mean(v):.4f}" + (f" (n={len(v)})" if len(v) > 1 else "")
                                     for g, v in lv.items()), flush=True)
    for a, b in (("human", "shuf"), ("human", "anti"), ("human", "rp")):
        ks = [k.split("_s")[1] for k in groups[a] if f"{b}_s{k.split('_s')[1]}" in ev]
        d = [f(ev[f"{a}_s{k}"]) - f(ev[f"{b}_s{k}"]) for k in ks]
        out["diffs"][f"{a}-{b}|{mname}"] = ci(d) + [len(d)]
        c = out["diffs"][f"{a}-{b}|{mname}"]
        print(f"  {a} − {b} (парно, n={len(d)}): {c[0]:+.4f} [{c[1]:+.4f}; {c[2]:+.4f}]", flush=True)
    if "base" in ev:
        for g in ("human", "shuf", "anti", "rp"):
            d = [v - f(ev["base"]) for v in lv[g]]
            out["diffs"][f"{g}-base|{mname}"] = ci(d) + [len(d)]
            c = out["diffs"][f"{g}-base|{mname}"]
            print(f"  {g} − base: {c[0]:+.4f} [{c[1]:+.4f}; {c[2]:+.4f}]", flush=True)
# правило П1 (experiments/PLAN_2026-11-11_amendments.md): годная пустышка — «случайная проекция» (rp)
rule = {}
pre = {m: (out["diffs"].get(f"human-base|{m}") or [None, -1, -1])[1] > 0 for m in ("tid2013 srocc", "kadid10k srocc")}
rule["предусловие human − base > 0"] = pre
per = {}
for m in ("tid2013 srocc", "kadid10k srocc"):
    hr, rb = out["diffs"].get(f"human-rp|{m}"), out["diffs"].get(f"rp-base|{m}")
    if not hr or hr[3] < 2:
        per[m] = "нет прогонов rp"
        continue
    if hr[1] > 0:
        per[m] = "прирост сверх годной пустышки"
    elif rb and rb[1] > 0 and hr[1] <= 0 <= hr[2]:
        per[m] = "прирост объясняется дообучением на согласованных с картинками сравнениях без суждений людей"
    else:
        per[m] = "не решено"
rule["по наборам"] = per
vals = set(per.values())
rule["итог"] = ("на одной ветви перенос не повторён" if not any(pre.values()) else
                (per["tid2013 srocc"] if len(vals) == 1 else "наборы расходятся — не решено"))
out["rule"] = rule
print("\nПРАВИЛО:", json.dumps(rule, ensure_ascii=False), flush=True)
json.dump(out, open(os.path.join(R, "summary.json"), "w"), ensure_ascii=False, indent=1)
print("SUMMARY_DONE")
