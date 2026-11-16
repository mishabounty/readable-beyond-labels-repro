#!/bin/bash
# Тезис 4 на других разбиениях эмоций A/B (задача «длина», обучаемый DistilBERT, 1 сид): r2 -- наименьшая
# информативность A о B, st1 -- наибольшая. task и unif концептов не видят: их z берутся из finetune_goemotions_length_v2.
set -e
cd "$(dirname "$0")"
PY=${PYTHON:-python}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=3
for c in r2 st1; do
  d=results/z/finetune_goemotions_length_${c}_v2
  mkdir -p $d
  for f in task_0_s0.npy unif_1_s0.npy; do ln -sf ../finetune_goemotions_length_v2/$f $d/$f; done
  $PY text_finetune.py --task length --concepts $c --seeds 0 --runs bce:1 rand:1 --out-tag _v2 --save-z
done
$PY probe_ridge.py finetune_goemotions_length_r2_v2 finetune_goemotions_length_st1_v2 --no-white
echo DONE
