#!/bin/bash
# Продолжение после сбоя: картиночный поток (лёгкий), затем методы против забывания на GoEmotions.
cd "$(dirname "$0")"
exec >> results/rev3_image_log.txt 2>&1
PY=${PYTHON:-python}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=3
date
$PY cub_finetune.py --dataset sun --concepts sun --seeds 1 2 --runs task:0 --out-tag _v2 --save-z --no-probe
$PY probe_ridge.py finetune_sun4_v2 finetune_sun4sem_v2 finetune_sun_v2 --no-white
$PY conditional_probe.py finetune_sun4_v2 finetune_sun4sem_v2 finetune_sun_v2
echo SUN_DONE; date
for K in 8 16 50; do
  $PY cub_finetune.py --dataset cub$K --seeds 0 1 2 --runs task:0 --out-tag _v2 --save-z --no-probe
done
$PY probe_ridge.py finetune_cub8_v2 finetune_cub16_v2 finetune_cub50_v2 --no-white
$PY conditional_probe.py finetune_cub8_v2 finetune_cub16_v2 finetune_cub50_v2
echo CUBK_DONE; date
$PY text_finetune.py --task length --seeds 0 1 2 --runs wise:0 lpft:0 l2sp:0.01 --out-tag _v2 --save-z
$PY probe_ridge.py finetune_goemotions_length_v2 --no-white
echo BASELINES_DONE; date
$PY robust_probe.py finetune_sun4_v2
echo IMAGE_DONE; date
