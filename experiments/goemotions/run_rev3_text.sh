#!/bin/bash
# Продолжение после сбоя 23.10 07:10 (OOM убил группу процессов приложения): текстовый поток.
cd "$(dirname "$0")"
exec >> results/rev3_text_log.txt 2>&1
PY=${PYTHON:-python}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=3
R="--dataset reuters --task length --max-len 128 --out-tag _v2 --save-z"
date
$PY text_finetune.py $R --seeds 1 2 --runs aonly:1 km:1
$PY text_finetune.py $R --seeds 2 3 4 --runs task:0 rand:1 bce:1
$PY probe_ridge.py finetune_reuters_length_v2 --no-white
$PY conditional_probe.py finetune_reuters_length_v2
$PY robust_probe.py finetune_reuters_length_v2
echo REUTERS_DONE; date
$PY text_finetune.py --dataset reuters --task country --max-len 128 --seeds 0 1 2 --runs task:0 rand:1 bce:1 --out-tag _v2 --save-z
$PY probe_ridge.py finetune_reuters_country_v2 --no-white
$PY conditional_probe.py finetune_reuters_country_v2
$PY robust_probe.py finetune_reuters_country_v2
echo COUNTRY_DONE; date
$PY text_finetune.py --task length --seeds 3 4 --runs hyp:1 recon:3 unif:1 --out-tag _v2 --save-z
$PY text_finetune.py $R --seeds 0 1 2 --runs hyp:1 recon:3 unif:1
$PY probe_ridge.py finetune_goemotions_length_v2 finetune_reuters_length_v2 --no-white
$PY conditional_probe.py finetune_goemotions_length_v2 finetune_reuters_length_v2
$PY robust_probe.py finetune_goemotions_length_v2 finetune_reuters_length_v2
echo EXTRAS_DONE; date
