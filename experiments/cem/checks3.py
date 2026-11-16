#!/usr/bin/env python
"""Опыты Э2, Э4, Э5 (план experiments/PLAN_2026-11-04_checks.md). Новые модели не обучаются.

  head — (GPU) собственные вероятности понятий c_sem у cemall_half и cem_half на всех фото test;
         файл features/<model>_fold<k>_head.npz;
  e2   — разницы placebo2 − hybrid, placebo − hybrid, cem − hybrid по probe_results.jsonl (acc_c1);
  e4   — логистическая проба вида по F, C по точности на val, мера — точность на test;
  e5   — на фото test видов S2: голова, блок из 16 чисел и проба по всему F; итог — results/.../checks3_e5.json.
Запуск: python checks3.py head; NJ=8 python checks3.py e2 e4 e5; SMOKE=1 — пробный (1 повтор, 6 атрибутов).
"""

import json
import os
import sys
import warnings

import numpy as np
from joblib import Parallel, delayed
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

sys.path.insert(0, os.path.expanduser("~/cem_repro"))
import analyze_nested as N  # noqa: E402

R = N.R
warnings.filterwarnings("ignore", category=ConvergenceWarning)
FD = N.FD
NJ = N.NJ
SMOKE = os.environ.get("SMOKE") == "1"
FOLDS = [0] if SMOKE else [0, 1, 2, 3, 4]
BLOCK = 16  # чисел на понятие в узком месте CEM
CS_BLOCK = [1e-3, 1e-2, 0.1, 1.0, 10.0]
CS_SPECIES = [1e-3, 1e-2, 0.1, 1.0]


def show(name, vals):
    """Среднее и границы d̄ ± t·s/√n по повторам."""
    m, lo, hi = N.interval(vals)
    print(f"  {name:38s}: {m:+.4f} [{lo:+.4f}; {hi:+.4f}]  по повторам {[round(float(v), 4) for v in vals]}", flush=True)
    return {"mean": m, "lo": lo, "hi": hi, "by_fold": [float(v) for v in vals]}


def mode_head():
    """Собственные вероятности понятий модели на test (выход 0 метода _forward)."""
    import torch
    import extract_features as E
    dl = E.P.loaders()["test"]
    for m in ("cemall_half", "cem_half"):
        for fold in FOLDS:
            f = os.path.join(FD, f"{m}_fold{fold}_head.npz")
            if os.path.exists(f):
                continue
            model = E.load_model(m, fold)
            P_, Y_ = [], []
            with torch.no_grad():
                for x, y, _ in dl:
                    P_.append(model._forward(x.to(E.P.DEV), train=False)[0].float().cpu()), Y_.append(y)
            head, Y = torch.cat(P_).numpy(), torch.cat(Y_).numpy().astype(np.int16)
            np.savez(f, head=head, Y=Y)
            print(f"сохранено {f}: {head.shape}, диапазон [{head.min():.3f}; {head.max():.3f}]", flush=True)
            del model
            torch.cuda.empty_cache()


def mode_e2():
    rows = [json.loads(s) for s in open(os.path.join(R.OUT, "probe_results.jsonl"))]
    val = {}
    for r in rows:
        if "acc_c1" in r:
            val[(r["model"], int(r["fold"]))] = 100.0 * float(r["acc_c1"]) if float(r["acc_c1"]) <= 1 else float(r["acc_c1"])
    res = {}
    print("Э2: точность пробы C = 1, п.", flush=True)
    for a in ("placebo2", "placebo", "cem"):
        res[f"{a}-hybrid"] = show(f"{a} − hybrid", [val[(a, k)] - val[("hybrid", k)] for k in FOLDS])
    json.dump(res, open(os.path.join(R.OUT, "checks3_e2.json"), "w"), indent=1)


def species_acc(F, feats, C):
    mu, sd = F["tr"].mean(0), F["tr"].std(0) + 1e-6
    m = LogisticRegression(C=C, max_iter=500).fit((F["tr"] - mu) / sd, feats["Y_train"].astype(int))
    return (float(m.score((F["va"] - mu) / sd, feats["Y_val"].astype(int))),
            100.0 * float(m.score((F["te"] - mu) / sd, feats["Y_test"].astype(int))))


def mode_e4():
    models = ["cem", "placebo", "placebo2", "cemnoc", "noconcept", "hybrid", "pretrained"]
    acc, own = {m: [] for m in models}, {m: [] for m in models}
    pre = np.load(os.path.join(FD, "pretrained_resnet34_imagenet_full.npz"))
    for fold in FOLDS:
        for m in models:
            z = pre if m == "pretrained" else np.load(os.path.join(FD, f"{m}_fold{fold}_full.npz"))
            feats = {k: z[k] for k in z.files}
            F = {"tr": feats["F_train"], "va": feats["F_val"], "te": feats["F_test"]}
            r = Parallel(n_jobs=min(NJ, len(CS_SPECIES)))(delayed(species_acc)(F, feats, C) for C in CS_SPECIES)
            best = int(np.argmax([x[0] for x in r]))
            acc[m].append(r[best][1])
            own[m].append(100.0 * float((feats["L_test"].argmax(1) == feats["Y_test"]).mean())
                          if "L_test" in feats else float("nan"))
            print(f"fold {fold} {m:10s}: проба вида {r[best][1]:.2f} (C = {CS_SPECIES[best]:g}), своя голова {own[m][-1]:.2f}", flush=True)
    print("Э4: разницы, точность вида, п.", flush=True)
    res = {"acc": acc, "own": own}
    for a, b in (("placebo2", "cemnoc"), ("placebo", "cemnoc"), ("cem", "cemnoc"), ("cemnoc", "noconcept"), ("cemnoc", "pretrained")):
        res[f"{a}-{b}"] = show(f"проба: {a} − {b}", np.array(acc[a]) - np.array(acc[b]))
    for m in models[:-1]:
        res[f"gap_{m}"] = show(f"проба − своя голова: {m}", np.array(acc[m]) - np.array(own[m]))
    json.dump(res, open(os.path.join(R.OUT, "checks3_e4.json"), "w"), indent=1)


def block_score(Ftr, ytr, groups, Fte, yte, j):
    """Проба понятия j по его блоку из 16 чисел; C по средней AP в GroupKFold по видам на S1."""
    a = Ftr[:, j * BLOCK:(j + 1) * BLOCK]
    b = Fte[:, j * BLOCK:(j + 1) * BLOCK]
    scores = []
    for C in CS_BLOCK:
        s = []
        for tr, va in GroupKFold(3).split(a, groups=groups):
            if ytr[tr].min() == ytr[tr].max() or ytr[va].sum() < 1:
                continue
            x, y = N.prep(a[tr], a[va])
            s.append(N.ap_score(ytr[va], N.fit_scores(x, ytr[tr], y, C)))
        scores.append(np.mean(s) if s else -1.0)
    C = CS_BLOCK[int(np.argmax(scores))]
    x, y = N.prep(a, b)
    return N.ap_score(yte, N.fit_scores(x, ytr, y, C))


def probe_full(F, ytr_mat, groups, Fte, Yte, keep):
    """Проба по всему F, подбор как в analyze_nested.py; возвращает AP по атрибутам."""
    best = N.tune(F, ytr_mat, groups)[0]
    return N.eval_attrs(F, ytr_mat, Fte, Yte, best[0], keep=keep)[1]


def mode_e5():
    T = N.species_table()
    pre = np.load(os.path.join(FD, "pretrained_resnet34_imagenet_full.npz"))
    out = {k: [] for k in ("head_B", "head_B_S1", "block_B", "base_B", "cem_half_B", "pre_B", "head_A", "head_all_A", "base_A", "n_B", "n_A")}
    for fold in FOLDS:
        sel = R.selected_concepts(fold)
        held = np.setdiff1d(np.arange(112), sel)
        if SMOKE:
            held = held[:6]
        sp = np.random.RandomState(100 + fold).permutation(200)
        S1, S2 = sp[:100], sp[100:]
        Y_tr, Y_te = pre["Y_train"].astype(int), pre["Y_test"].astype(int)
        C_tr, C_te = pre["C_train"].astype(int), pre["C_test"].astype(int)
        itr, ite, ite1 = np.isin(Y_tr, S1), np.isin(Y_te, S2), np.isin(Y_te, S1)
        keep_of = lambda ids: (T[S1][:, ids].sum(0) >= N.MIN_SP) & (T[S2][:, ids].sum(0) >= N.MIN_SP)  # noqa: E731
        keepB, keepA = keep_of(held), keep_of(sel)
        Btr, Bte, Bte1 = C_tr[itr][:, held], C_te[ite][:, held], C_te[ite1][:, held]
        Atr, Ate = C_tr[itr][:, sel], C_te[ite][:, sel]

        h_all = np.load(os.path.join(FD, f"cemall_half_fold{fold}_head.npz"))
        h_cem = np.load(os.path.join(FD, f"cem_half_fold{fold}_head.npz"))
        assert np.array_equal(h_all["Y"].astype(int), Y_te) and np.array_equal(h_cem["Y"].astype(int), Y_te)
        assert h_all["head"].shape[1] == 112 and h_cem["head"].shape[1] == len(sel)

        def ap_cols(Y, sc, keep):
            return np.array([N.ap_score(Y[:, j], sc[:, j]) if keep[j] and Y[:, j].sum() > 0 else np.nan for j in range(Y.shape[1])])

        apB = ap_cols(Bte, h_all["head"][ite][:, held], keepB)
        apB1 = ap_cols(Bte1, h_all["head"][ite1][:, held], keepB)
        apA_cem = ap_cols(Ate, h_cem["head"][ite], keepA)
        apA_all = ap_cols(Ate, h_all["head"][ite][:, sel], keepA)

        z_all = np.load(os.path.join(FD, f"cemall_half_fold{fold}_full.npz"))
        Fa_tr, Fa_te = z_all["F_train"][itr], z_all["F_test"][ite]
        idx = [j for j in range(len(held)) if keepB[j]]
        vals = Parallel(n_jobs=NJ)(delayed(block_score)(Fa_tr, Btr[:, j], Y_tr[itr], Fa_te, Bte[:, j], int(held[j])) for j in idx)
        apBlock = np.full(len(held), np.nan)
        apBlock[idx] = vals

        z_noc = np.load(os.path.join(FD, f"cemnoc_half_fold{fold}_full.npz"))
        z_cem = np.load(os.path.join(FD, f"cem_half_fold{fold}_full.npz"))
        base_B = probe_full(z_noc["F_train"][itr], Btr, Y_tr[itr], z_noc["F_test"][ite], Bte, keepB)
        cem_B = probe_full(z_cem["F_train"][itr], Btr, Y_tr[itr], z_cem["F_test"][ite], Bte, keepB)
        pre_B = probe_full(pre["F_train"][itr], Btr, Y_tr[itr], pre["F_test"][ite], Bte, keepB)
        base_A = probe_full(z_noc["F_train"][itr], Atr, Y_tr[itr], z_noc["F_test"][ite], Ate, keepA)

        row = dict(head_B=apB, head_B_S1=apB1, block_B=apBlock, base_B=base_B, cem_half_B=cem_B, pre_B=pre_B,
                   head_A=apA_cem, head_all_A=apA_all, base_A=base_A)
        for k, v in row.items():
            out[k].append(float(np.nanmean(v)))
        out["n_B"].append(int(keepB.sum()))
        out["n_A"].append(int(keepA.sum()))
        print(f"fold {fold}: " + ", ".join(f"{k} {np.nanmean(v):.4f}" for k, v in row.items()) + f"; B {keepB.sum()}, A {keepA.sum()}", flush=True)
    print("Э5: разницы AP на видах S2", flush=True)
    a = {k: np.array(v) for k, v in out.items()}
    res = {"raw": out}
    res["5a"] = show("Э5а голова cemall B − F cemnoc B", a["head_B"] - a["base_B"])
    res["5b"] = show("Э5б блок cemall B − F cemnoc B", a["block_B"] - a["base_B"])
    res["5c"] = show("Э5в голова cem_half A − F cemnoc A", a["head_A"] - a["base_A"])
    res["5c_all"] = show("голова cemall A − F cemnoc A", a["head_all_A"] - a["base_A"])
    res["gap"] = show("голова cemall B: S1 − S2", a["head_B_S1"] - a["head_B"])
    res["base_pre"] = show("F cemnoc B − F pretrained B", a["base_B"] - a["pre_B"])
    res["cem_base"] = show("F cem_half B − F cemnoc B", a["cem_half_B"] - a["base_B"])
    json.dump(res, open(os.path.join(R.OUT, "checks3_e5.json"), "w"), indent=1)


if __name__ == "__main__":
    for mode in sys.argv[1:]:
        {"head": mode_head, "e2": mode_e2, "e4": mode_e4, "e5": mode_e5}[mode]()
    print("CHECKS3_DONE", flush=True)
