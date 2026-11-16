#!/bin/bash
# Доработка: дообучаемый DistilBERT (прогоны v2, прерванные нехваткой памяти GPU) -- после CUB-разбиений и Cora.
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=${PYTHON:-python}
cd "$(dirname "$0")"
until grep -q REV_CORA_DONE results/rev_cubsplits_log.txt; do sleep 60; done
$PY text_finetune.py --seeds 0 --runs rank:1 rand:1 recon:3 unif:1 --out-tag _v2 --save-z
$PY text_finetune.py --seeds 1 --runs task:0 bce:1 rank:1 rand:1 recon:3 unif:1 --out-tag _v2 --save-z
echo REV_TEXTFT_DONE
