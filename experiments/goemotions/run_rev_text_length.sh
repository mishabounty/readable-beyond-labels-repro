#!/bin/bash
# Доработка: обучаемый DistilBERT с задачей, не связанной с эмоциями (квартиль длины текста) -- текстовый аналог
# случайных суперклассов CUB-4. После обучаемых CUB (память GPU).
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=${PYTHON:-python}
cd "$(dirname "$0")"
until grep -q REV_FTMORE_DONE results/rev_ft_more_log.txt 2>/dev/null; do sleep 60; done
$PY text_finetune.py --task length --seeds 0 1 --runs task:0 bce:1 rank:1 rand:1 recon:3 unif:1 --out-tag _v2 --save-z
$PY probe_ridge.py finetune_goemotions_length_v2 --no-white
echo REV_TEXTLEN_DONE
