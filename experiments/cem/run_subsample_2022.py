#!/usr/bin/env python
"""Повторение опыта CEM §5.3 / A.9 (Espinosa Zarlenga et al., NeurIPS 2022) кодом авторов, коммит 26f027dd6f0c.

Запускает одну модель на одном разбиении (fold) с настройками из experiments/cub_subsample_experiment.py того же
коммита, только для доли понятий 0.25 (28 из 112). Скрипт авторов как есть не запускается (неверные пути
импорта, не определён CUB_DIR), поэтому здесь те же вызовы training.train_model с тем же og_config и теми же
конфигурациями моделей; логика опыта не меняется.
  noconcept -- NoConceptSupervisionReLU_ExtraCapacity (та же ёмкость, вес понятий 0), как у авторов;
  hybrid    -- FuzzyExtraCapacity_Logit (Hybrid-CBM), как у авторов;
  cem       -- MixtureEmbModel (CEM, RandInt 0.25), как у авторов;
  placebo   -- тот же CEM, но метки 28 понятий заменены пустышкой: у каждого понятия свои метки, переставленные
               между видами (make_placebo.py); частоты и структура «по видам» те же, смысла нет;
  placebo2  -- тот же CEM на пустышке-2: вид получает целиком вектор атрибутов другого вида (make_placebo2.py);
  placebo3  -- тот же CEM на пустышке, зависящей от входа: случайные плоскости в признаках исходной ResNet (make_placebo3.py);
  cemnoc    -- та же архитектура CEM без понятий: вес понятий 0 и RandInt выключен (иначе настоящие метки
               попадали бы в узкое место); отделяет вклад устройства CEM от вклада меток уровня вида.
Отличие от авторов: 28 понятий каждого разбиения выбираются np.random.RandomState(1000 + fold), а не цепочкой
вызовов их скрипта (их выбор зависит от всей последовательности прогонов); выбор сохраняется в файл.
"""

import argparse
import copy
import json
import os
import sys

import numpy as np

REPO = os.path.expanduser("~/cem_repro/cem")
sys.path.insert(0, REPO)
import cem.train.training as training  # noqa: E402
from cem.data.CUB200 import cub_loader  # noqa: E402

CUB_DIR = os.path.join(REPO, "cem", "data", "CUB200")
BASE_DIR = os.path.join(CUB_DIR, "class_attr_data_10")
OUT = os.environ.get("CEM_OUT", os.path.expanduser("~/cem_repro/results/cub_subsample_0.25"))
N_CONCEPTS, N_TASKS, PERCENT = 112, 200, 0.25

OG_CONFIG = dict(  # дословно og_config из cub_subsample_experiment.py (26f027dd6f0c)
    cv=5, max_epochs=300, patience=15, batch_size=128, num_workers=8, emb_size=16, extra_dims=0,
    concept_loss_weight=5, normalize_loss=False, learning_rate=0.01, weight_decay=4e-05, scheduler_step=20,
    weight_loss=True, pretrain_model=True, c_extractor_arch="resnet34", optimizer="sgd", bool=False,
    early_stopping_monitor="val_loss", early_stopping_mode="min", early_stopping_delta=0.0, corr_thresh=0.5,
    dense_corr_thresh=0.25, sampling_percent=1, sampling_percents=[0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 1],
    momentum=0.9, shared_prob_gen=False, sigmoidal_prob=False, sigmoidal_embedding=False,
    training_intervention_prob=0.0, embeding_activation=None, concat_prob=False,
)


# модели, обученные только на половине видов S1 = RandomState(100 + fold).permutation(200)[:100] (make_half_species.py):
# имя -> (базовая конфигурация, папка данных)
HALF = {"cem_half": ("cem", "half_data"), "placebo2_half": ("placebo2", "placebo2_half_data"),
        "cemnoc_half": ("cemnoc", "half_data"), "cemall_half": ("cem", "half_data")}
# cemall_half (план experiments/PLAN_2026-11-02_checks.md, N3) — положительный контроль опыта C: та же CEM на видах
# S1, но обученная на всех 112 атрибутах (A и B); проверяет, способна ли постановка показать перенос обученных понятий
# на виды, которых модель не видела.


def data_dir_for(name, fold):
    """Папка train/val/test.pkl, на которых училась модель (для поправки на дисбаланс и для запуска)."""
    if name in ("placebo", "placebo2", "placebo3", "placebo4"):
        return os.path.join(OUT, f"{name}_data", f"fold_{fold}")
    if name == "placebo4w":  # те же коды, что placebo4
        return os.path.join(OUT, "placebo4_data", f"fold_{fold}")
    if name in HALF:
        return os.path.join(OUT, HALF[name][1], f"fold_{fold}")
    return BASE_DIR


def imbalance_pkl(name, fold):
    """Откуда считать веса потерь понятий (поправка на дисбаланс авторов): обычно из train.pkl той же папки, на которой
    училась модель; у placebo4w — из настоящих меток Koh (план experiments/PLAN_2026-11-11_amendments.md, П2)."""
    if name == "placebo4w":
        return os.path.join(BASE_DIR, "train.pkl")
    return os.path.join(data_dir_for(name, fold), "train.pkl")


def model_config(name, k):
    """Конфигурации моделей дословно из cub_subsample_experiment.py (26f027dd6f0c)."""
    if name in HALF:
        c = model_config(HALF[name][0], k)
        c["extra_name"] = ("HalfAll_" if name == "cemall_half" else "Half_") + c["extra_name"]
        return c
    c = copy.deepcopy(OG_CONFIG)
    c["sampling_percent"] = PERCENT
    if name == "noconcept":
        c.update(architecture="ConceptBottleneckModel", bool=False, extra_dims=c["emb_size"] * k,
                 bottleneck_nonlinear="relu", concept_loss_weight=0,
                 extra_name=f"NoConceptSupervisionReLU_ExtraCapacity_subsample_{PERCENT}")
    elif name == "hybrid":
        c.update(architecture="ConceptBottleneckModel", bool=False, extra_dims=c["emb_size"] * k,
                 sigmoidal_extra_capacity=False, sigmoidal_prob=False,
                 extra_name=f"FuzzyExtraCapacity_Logit_subsample_{PERCENT}")
    elif name == "hybrid448":
        # Hybrid как в тексте статьи CEM (§5.1, Baselines): в узком месте k·m чисел, как у CEM, Leaky-ReLU на свободных
        # (план experiments/PLAN_2026-11-10_checks.md, Г1). В cbm.py нелинейность к свободным числам применяется только
        # при sigmoidal_prob=True: тогда 28 понятий — вероятности, 420 свободных — Leaky-ReLU, всего 448.
        c.update(architecture="ConceptBottleneckModel", bool=False, extra_dims=c["emb_size"] * k - k,
                 sigmoidal_extra_capacity=False, sigmoidal_prob=True, bottleneck_nonlinear="leakyrelu",
                 extra_name=f"Paper448LeakyReLU_subsample_{PERCENT}")
    elif name in ("cem", "placebo", "placebo2", "placebo3", "placebo4", "placebo4w", "cemnoc"):
        c.update(architecture="MixtureEmbModel", shared_prob_gen=True, sigmoidal_prob=True,
                 sigmoidal_embedding=False, training_intervention_prob=0.25, concat_prob=False,
                 embeding_activation="leakyrelu",
                 extra_name=f"{ {'placebo': 'Placebo_', 'placebo2': 'Placebo2_', 'placebo3': 'Placebo3_', 'placebo4': 'Placebo4_', 'placebo4w': 'Placebo4W_', 'cemnoc': 'NoConcepts_'}.get(name, '')}"
                            f"SharedProb_AdaptiveDropout_NoProbConcat_subsample_{PERCENT}")
        if name == "cemnoc":
            c.update(concept_loss_weight=0, training_intervention_prob=0.0)
    else:
        raise ValueError(name)
    return c


def selected_concepts(fold):
    f = os.path.join(OUT, f"selected_concepts_sampling_{PERCENT}_fold_{fold}.npy")
    if os.path.exists(f):
        return np.load(f)
    sel = np.random.RandomState(1000 + fold).permutation(N_CONCEPTS)[:int(np.ceil(N_CONCEPTS * PERCENT))]
    np.save(f, sel)
    return sel


def concepts_for(name, fold):
    """Понятия, на которых учится модель: все 112 у cemall_half, иначе 28 выбранных для повтора."""
    return np.arange(N_CONCEPTS) if name == "cemall_half" else selected_concepts(fold)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["noconcept", "hybrid", "hybrid448", "cem", "placebo", "placebo2", "placebo3", "placebo4", "placebo4w", "cemnoc"] + list(HALF))
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--num-workers", type=int, default=3)
    ap.add_argument("--max-epochs", type=int, default=0, help="> 0 -- только для пробного прогона")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    selected_concepts(args.fold)  # файл выбора 28 понятий повтора должен существовать и для cemall_half
    sel = concepts_for(args.model, args.fold)
    k = len(sel)
    data_dir = data_dir_for(args.model, args.fold)
    train_p, val_p, test_p = (os.path.join(data_dir, f"{s}.pkl") for s in ("train", "val", "test"))
    imbalance = np.array(cub_loader.find_class_imbalance(imbalance_pkl(args.model, args.fold), True))[sel]

    def subsample_transform(sample):
        if isinstance(sample, list):
            sample = np.array(sample)
        return sample[sel]

    config = model_config(args.model, k)
    config["num_workers"] = args.num_workers
    if args.max_epochs:
        config["max_epochs"] = args.max_epochs
    dls = [cub_loader.load_data(pkl_paths=[p], use_attr=True, no_img=False, batch_size=config["batch_size"],
                                uncertain_label=False, n_class_attr=2, image_dir="images", resampling=False,
                                root_dir=CUB_DIR, num_workers=config["num_workers"],
                                concept_transform=subsample_transform) for p in (train_p, val_p, test_p)]
    model, test_results = training.train_model(
        n_concepts=k, n_tasks=N_TASKS, config=config, train_dl=dls[0], val_dl=dls[1], test_dl=dls[2],
        split=args.fold, result_dir=OUT, rerun=False, project_name="", seed=args.fold,
        activation_freq=0, single_frequency_epochs=0, imbalance=imbalance)
    res = {kk: float(v) if np.isscalar(v) else str(v) for kk, v in (test_results or {}).items()}
    res.update(model=args.model, fold=args.fold, selected=sel.tolist())
    with open(os.path.join(OUT, f"test_results_{args.model}_fold_{args.fold}.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    print("RESULT", json.dumps({kk: v for kk, v in res.items() if kk != "selected"}), flush=True)


if __name__ == "__main__":
    main()
