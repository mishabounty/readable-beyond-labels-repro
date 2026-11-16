#!/bin/bash
# Тезис 4 на втором наборе: Reuters-21578, концепты -- темы новостей (A 19 / B 20), задача -- квартиль длины,
# обучаемые 2 верхних слоя DistilBERT, 128 токенов, 2 сида (минимальный прогон).
# Может изменить: общность тезиса 4. Предсказание: task < вход (потеря информации), bce - rand > 0 (+0.02...+0.07).
# Если bce - rand <= 0 на обоих сидах -- тезис 4 сузить до эмоций; если task >= входа -- режим не достигнут.
set -e
cd "$(dirname "$0")"
PY=${PYTHON:-python}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=3
$PY text_finetune.py --dataset reuters --task length --max-len 128 --seeds 0 1 --runs task:0 bce:1 rand:1 --out-tag _v2 --save-z
$PY probe_ridge.py finetune_reuters_length_v2 --no-white
$PY conditional_probe.py finetune_reuters_length_v2
echo DONE
