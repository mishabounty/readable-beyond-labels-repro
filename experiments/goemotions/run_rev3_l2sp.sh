#!/bin/bash
# L2-SP с более сильным штрафом (0.01 почти не отличался от task-only) -- после картиночного потока.
cd "$(dirname "$0")"
exec >> results/rev3_l2sp_log.txt 2>&1
PY=${PYTHON:-python}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=3
until grep -q IMAGE_DONE results/rev3_image_log.txt 2>/dev/null; do sleep 30; done
date
$PY text_finetune.py --task length --seeds 0 1 2 --runs l2sp:0.1 l2sp:1 --out-tag _v2 --save-z
$PY probe_ridge.py finetune_goemotions_length_v2 --no-white
echo L2SP_DONE; date
