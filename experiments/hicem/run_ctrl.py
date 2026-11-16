#!/usr/bin/env python
"""Контрольные варианты опыта HiCEM (Hill et al., ICLR 2026) на их коде (коммит 4b665fa), без правки их файлов.

Запуск из ~/hicem_repro/code/experiments:
  python run_ctrl.py <config.yaml> true_filter        — Rtrue: настоящая CEM, но фото для поиска подпонятий отбираются
                                                         по НАСТОЯЩИМ грубым меткам (как в C1), а не по ответу CEM;
  python run_ctrl.py <config.yaml> null_bank <seed>    — Rnull: опыт R без изменений (отбор фото по ответу CEM, как
                                                         у авторов), но банк тонких понятий заменён случайным: метки
                                                         каждого понятия банка перемешаны между видами внутри видов с
                                                         родительским грубым понятием = 1 (одинаково для train и test).
                                                         Показывает, что даёт их мера (табл. 1) для бессмысленных
                                                         «подпонятий» — избирательность меры;
  python run_ctrl.py <config.yaml> placebo <seed>      — P: начальная CEM учится на пустышке — грубые понятия
                                                         перетасованы между видами одной перестановкой (вид получает
                                                         весь набор ответов другого вида, без неподвижных точек);
                                                         отбор фото — по настоящим меткам, итоговая HiCEM учится на
                                                         настоящих грубых метках и найденных подпонятиях.
Так R → Rtrue → P отличаются по одному фактору: Rtrue − R — способ отбора фото; P − Rtrue — смысл грубых понятий,
на которых учится CEM; C1 (поиск прямо на признаках CLIP) отбирает фото так же, как Rtrue и P.
"""

import copy
import sys
from pathlib import Path

import numpy as np
import torch

import cemcd.concept_discovery as CD
import run_experiment as RE
from experiment_utils import load_config

cfg_path, mode = sys.argv[1], sys.argv[2]
seed = int(sys.argv[3]) if len(sys.argv) > 3 else None
assert mode in ("true_filter", "placebo", "null_bank")

# отбор фото для поиска подпонятий по настоящим грубым меткам (в split_concepts predictions берутся отсюда)
_orig_embeddings = CD.calculate_embeddings


def _embeddings_true_filter(model, dl):
    c_pred, c_embs, y_pred = _orig_embeddings(model, dl)
    c_true = dl.dataset.tensors[2].numpy()
    assert c_true.shape == c_pred.shape, (c_true.shape, c_pred.shape)
    return c_true, c_embs, y_pred


if mode in ("true_filter", "placebo"):
    _orig_split = CD.split_concepts

    def _split_true_filter(*a, **k):
        # подмена действует только на время поиска подпонятий: их код вызывает calculate_embeddings и позже,
        # и постоянная подмена роняла последний вспомогательный шаг (исправлено 30.10)
        CD.calculate_embeddings = _embeddings_true_filter
        try:
            return _orig_split(*a, **k)
        finally:
            CD.calculate_embeddings = _orig_embeddings

    CD.split_concepts = _split_true_filter

if mode == "null_bank":
    _orig_load = RE.load_datasets

    def _null_bank_load(config):
        datasets = _orig_load(config)
        rng = np.random.RandomState(seed)
        for ds in datasets:
            ytr, yte = ds.train_y.numpy().astype(int), ds.test_y.numpy().astype(int)
            c = ds.train_c.numpy()
            first = np.array([np.where(ytr == k)[0][0] for k in range(ds.n_tasks)])
            bank_cls, coarse_cls = ds.concept_bank[first].copy(), c[first]
            assert np.array_equal(ds.concept_bank, bank_cls[ytr])
            parent = {}
            for top, members in enumerate(ds.sub_concept_map):
                for j in members:
                    parent.setdefault(j, top)
            bnull = bank_cls.copy()
            for j in range(bank_cls.shape[1]):
                pool = np.where(coarse_cls[:, parent[j]] == 1)[0] if j in parent else np.arange(ds.n_tasks)
                bnull[pool, j] = bank_cls[rng.permutation(pool), j]
            ds.concept_bank = bnull[ytr]
            ds.concept_test_ground_truth = bnull[yte]
            print(f"случайный банк: совпадение с настоящим {float((bnull == bank_cls).mean()):.3f}", flush=True)
        return datasets

    RE.load_datasets = _null_bank_load

if mode == "placebo":
    _orig_train = RE.train_initial_cems

    def _placebo_train(config, datasets, run_dir):
        placebo_sets = []
        for ds in datasets:
            y, c = ds.train_y.numpy(), ds.train_c.numpy()
            table = np.zeros((ds.n_tasks, c.shape[1]), dtype=np.float32)
            for k in range(ds.n_tasks):  # грубые понятия заданы на уровне вида
                rows = c[y == k]
                assert len(rows) and np.all(rows == rows[0]), f"грубые понятия вида {k} не постоянны"
                table[k] = rows[0]
            rng = np.random.RandomState(seed)
            while True:
                pi = rng.permutation(ds.n_tasks)
                if not np.any(pi == np.arange(ds.n_tasks)):
                    break
            p = copy.copy(ds)
            for split in ("train", "val", "test"):
                ys = getattr(ds, f"{split}_y").numpy()
                setattr(p, f"{split}_c", torch.from_numpy(table[pi[ys]]))
            same = float((table[pi] == table).mean())
            print(f"пустышка грубых понятий: совпадение с настоящими {same:.3f}; "
                  f"разных наборов: настоящие {len({tuple(r) for r in table})}, "
                  f"пустышка {len({tuple(r) for r in table[pi]})}", flush=True)
            placebo_sets.append(p)
        return _orig_train(config, placebo_sets, run_dir)

    RE.train_initial_cems = _placebo_train

if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    config = load_config(cfg_path)
    run_name = RE.create_run_name(config["results_dir"], config["dataset"])
    run_dir = Path(config["results_dir"]) / run_name
    run_dir.mkdir(parents=True)
    (run_dir / "config.yaml").write_text(Path(cfg_path).read_text() + f"\n# run_ctrl.py: mode={mode} seed={seed}\n")
    print(f"RUN NAME: {run_name}; mode {mode}, seed {seed}", flush=True)
    RE.run_experiment(run_dir=run_dir, config=config)
    print("CTRL_DONE", flush=True)
