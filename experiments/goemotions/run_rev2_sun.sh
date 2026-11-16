#!/bin/bash
# SUN Attributes параллельно с текстовыми очередями (лёгкое обучение layer4).
cd "$(dirname "$0")"
PY=${PYTHON:-python}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=3
until grep -q "features saved" results/sun_prepare_log.txt 2>/dev/null; do sleep 60; done
# 1. SUN Attributes: случайные суперклассы (несвязанная задача) -- полный набор целей; остальные задачи -- task
$PY cub_finetune.py --dataset sun4 --concepts sun --seeds 0 1 2 --runs task:0 rand:1 bce:1 unif:1 --out-tag _v2 --save-z --no-probe
$PY cub_finetune.py --dataset sun4sem --concepts sun --seeds 0 1 2 --runs task:0 --out-tag _v2 --save-z --no-probe
$PY cub_finetune.py --dataset sun --concepts sun --seeds 0 1 2 --runs task:0 --out-tag _v2 --save-z --no-probe
$PY probe_ridge.py finetune_sun4_v2 finetune_sun4sem_v2 finetune_sun_v2 --no-white
$PY conditional_probe.py finetune_sun4_v2 finetune_sun4sem_v2 finetune_sun_v2
echo SUN_DONE
