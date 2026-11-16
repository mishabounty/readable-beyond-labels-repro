#!/bin/bash
# Доработка, третья очередь (после дообучаемых прогонов): ещё 4 случайных разбиения групп атрибутов CUB.
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3
PY=${PYTHON:-python}
cd "$(dirname "$0")"
until grep -q FT_DONE results/v2_ft_log.txt; do sleep 60; done
for s in 1 2 3 4; do
  for ds in cub4 cub; do
    $PY collapse_check.py --datasets $ds --train-dict cub_g${s}A --eval-dict cub_g${s}B --k 0 --emb inception --seeds 0 1 \
      --runs task:0 bce:1 bce_bal:1 rank:1 rand:1 recon:3 unif:1 vicreg:1 cbm:1 --out-tag _gsplit$s --save-z --no-raw --no-probe
  done
done
echo REV_CUBSPLITS_DONE
# Cora: ещё 3 разбиения основ слов (seed 123 -- основное), 2 seed обучения
cd ../EmbeddedCodebookInterpretability/CodebookReviewChecks
for ss in 1 2 3; do
  $PY cora_collapse.py --split-seed $ss --seeds 0 1 --save-z
  $PY cora_collapse.py --split-seed $ss --seeds 0 --input-reference
done
echo REV_CORA_DONE
