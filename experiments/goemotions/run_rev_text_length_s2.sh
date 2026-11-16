#!/bin/bash
# Третий сид для обучаемого DistilBERT с задачей «длина текста» (ключевой результат; rank близок к bce -- без него).
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=${PYTHON:-python}
cd "$(dirname "$0")"
$PY text_finetune.py --task length --seeds 2 --runs task:0 bce:1 rand:1 unif:1 --out-tag _v2 --save-z
$PY probe_ridge.py finetune_goemotions_length_v2 --no-white
echo REV_TEXTLEN_S2_DONE
