#!/bin/bash
# Доработка: обучаемый ResNet-18 layer4 при разной связанности задачи с атрибутами: 4 суперкласса по атрибутам и
# 200 видов (к случайным суперклассам CUB-4 из прогонов v2); затем VICReg и bce_bal для CUB-4.
# После дообучения DistilBERT (память GPU).
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=${PYTHON:-python}
cd "$(dirname "$0")"
until grep -q REV_TEXTFT_DONE results/rev_text_ft_log.txt 2>/dev/null; do sleep 60; done
for ds in cub4sem cub; do
  $PY cub_finetune.py --dataset $ds --seeds 0 1 2 --runs task:0 bce:1 rank:1 rand:1 recon:3 unif:1 \
    --out-tag _v2 --save-z --no-probe
done
$PY cub_finetune.py --dataset cub4 --seeds 0 1 2 --runs vicreg:1 bce_bal:1 --out-tag _v2 --save-z --no-probe
for ds in cub4sem cub; do
  $PY cub_finetune.py --dataset $ds --seeds 0 1 2 --runs vicreg:1 bce_bal:1 --out-tag _v2 --save-z --no-probe
done
echo REV_FTMORE_DONE
