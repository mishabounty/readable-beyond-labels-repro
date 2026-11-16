#!/bin/bash
# Быстрая проверка опыта CEM §5.3: 4 модели x 2 запуска, затем чтение (cem_readout.py)
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=3
PY=${PYTHON:-python}
$PY cem_check.py --seeds 0 1 --runs task:0:cemA cem:1:cemA cem:1:cemArnd hcbm:1:cemA
echo "TRAIN DONE"
