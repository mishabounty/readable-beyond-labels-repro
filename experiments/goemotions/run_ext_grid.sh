#!/bin/bash
# 26.10: пересчёт проб с расширенной сеткой штрафа ridge (до 1e-8 * mean eigenvalue) на сохранённых z.
# Новые результаты -- в файлах *_ext_*; исходные результаты не меняются. Обучения нет.
set -u
cd "$(dirname "$0")"
PY=${PYTHON:-python}
export RIDGE_GRID=ext
FROZEN="cebab_v2 goemotions_v2 goemotions_humanAB_v2 cub_AB_v2 cub4_AB_v2 cub4sem_AB_v2 goemotions_humanAB_tfidf"
TUNED="finetune_goemotions_length_v2 finetune_reuters_length_v2 finetune_reuters_country_v2 finetune_sun4_v2 \
finetune_cub4_v2 finetune_cub4sem_v2 finetune_cub_v2 finetune_goemotions_v2 finetune_cub8_v2 finetune_cub16_v2 \
finetune_cub50_v2 finetune_sun4sem_v2 finetune_sun_v2 finetune_goemotions_length_r2_v2 finetune_goemotions_length_st1_v2"
echo "== probe_ridge $(date)"; $PY probe_ridge.py --no-white $TUNED $FROZEN
echo "== conditional $(date)"; $PY conditional_probe.py finetune_goemotions_length_v2 finetune_reuters_length_v2 \
  finetune_reuters_country_v2 finetune_sun4_v2 finetune_cub4_v2 finetune_cub4sem_v2 finetune_cub_v2 finetune_goemotions_v2 \
  finetune_cub8_v2 finetune_cub16_v2 finetune_cub50_v2 finetune_sun4sem_v2 finetune_sun_v2
echo "== robust $(date)"; $PY robust_probe.py finetune_goemotions_length_v2 finetune_reuters_length_v2 \
  finetune_reuters_country_v2 finetune_sun4_v2 --skip mlp mlpw
echo "== budget $(date)"; $PY probe_budget.py cebab_v2 finetune_cub4_v2
echo "== done $(date)"
