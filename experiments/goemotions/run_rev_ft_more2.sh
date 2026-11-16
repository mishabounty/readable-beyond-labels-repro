#!/bin/bash
# Продолжение run_rev_ft_more.sh после остановки сессии: VICReg и bce_bal для обучаемых CUB (cub4 seed 0 уже есть).
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=${PYTHON:-python}
cd "$(dirname "$0")"
$PY cub_finetune.py --dataset cub4 --seeds 1 2 --runs vicreg:1 bce_bal:1 --out-tag _v2 --save-z --no-probe
for ds in cub4sem cub; do
  $PY cub_finetune.py --dataset $ds --seeds 0 1 2 --runs vicreg:1 bce_bal:1 --out-tag _v2 --save-z --no-probe
done
echo REV_FTMORE_DONE >> results/rev_ft_more_log.txt
