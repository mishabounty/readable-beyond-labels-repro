#!/bin/bash
# 26.10: продолжение run_ext_grid.sh после OOM на TF-IDF: условная проба, проверки, бюджет; TF-IDF -- последним.
set -u
cd "$(dirname "$0")"
PY=${PYTHON:-python}
export RIDGE_GRID=ext
echo "== conditional $(date)"; $PY conditional_probe.py finetune_goemotions_length_v2 finetune_reuters_length_v2 \
  finetune_reuters_country_v2 finetune_sun4_v2 finetune_cub4_v2 finetune_cub4sem_v2 finetune_cub_v2 finetune_goemotions_v2 \
  finetune_cub8_v2 finetune_cub16_v2 finetune_cub50_v2 finetune_sun4sem_v2 finetune_sun_v2
echo "== robust $(date)"; $PY robust_probe.py finetune_goemotions_length_v2 finetune_reuters_length_v2 \
  finetune_reuters_country_v2 finetune_sun4_v2 --skip mlp mlpw
echo "== budget $(date)"; $PY probe_budget.py cebab_v2 finetune_cub4_v2
echo "== tfidf $(date)"; $PY probe_ridge.py --no-white goemotions_humanAB_tfidf
echo "== done $(date)"
