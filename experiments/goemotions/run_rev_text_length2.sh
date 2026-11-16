#!/bin/bash
# Обучаемый DistilBERT, задача не связана с эмоциями (квартиль длины текста). Без recon (дублирует unif).
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=${PYTHON:-python}
cd "$(dirname "$0")"
while kill -0 141012 2>/dev/null; do sleep 30; done   # VICReg/bce_bal CUB-4 (память GPU)
$PY probe_ridge.py finetune_cub4_v2 --no-white
$PY text_finetune.py --task length --seeds 0 1 --runs task:0 bce:1 rank:1 rand:1 unif:1 --out-tag _v2 --save-z
$PY probe_ridge.py finetune_goemotions_length_v2 --no-white
echo REV_TEXTLEN_DONE
