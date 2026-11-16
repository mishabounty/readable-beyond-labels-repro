#!/bin/bash
# Доработка 2 (рецензия v2), фаза 1: тезис 4 против контролей «только A» и псевдо-концептов k-means, 5 сидов.
# Может сломать тезис 4: если aonly или km >= bce, смысл концептов не нужен (перенос от любой структурированной цели).
set -e
cd "$(dirname "$0")"
PY=${PYTHON:-python}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=3
G="--task length --out-tag _v2 --save-z"
R="--dataset reuters --task length --max-len 128 --out-tag _v2 --save-z"
$PY text_finetune.py $G --seeds 0 1 2 3 4 --runs aonly:1 km:1
$PY text_finetune.py $G --seeds 3 4 --runs task:0 rand:1 bce:1
$PY probe_ridge.py finetune_goemotions_length_v2 --no-white
echo PHASE1_GOEMO_DONE
$PY text_finetune.py $R --seeds 0 1 2 --runs aonly:1 km:1
$PY text_finetune.py $R --seeds 2 3 4 --runs task:0 rand:1 bce:1
$PY probe_ridge.py finetune_reuters_length_v2 --no-white
echo PHASE1_DONE
