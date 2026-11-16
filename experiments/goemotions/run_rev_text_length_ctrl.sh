#!/bin/bash
# Задача «длина текста», обучаемый DistilBERT: контроли без смысла сильнее rand --
# hyp (случайные гиперплоскости входа, доли как у эмоций A) и recon (привязка к исходному эмбеддингу, аналог LDIFS).
set -e
cd "$(dirname "$0")"
PY=${PYTHON:-python}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=3
$PY text_finetune.py --task length --seeds 0 1 2 --runs hyp:1 recon:3 --out-tag _v2 --save-z
$PY probe_ridge.py finetune_goemotions_length_v2 --no-white
echo DONE
