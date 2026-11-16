#!/bin/bash
# D2 с направлением концепта «разность средних» (не зависит от обучения пробы) и доделки прогонов, упавших из-за
# нехватки памяти GPU. Стартует после D2 из run_rev_grid.sh (когда та перешла к ручкам схлопывания).
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=${PYTHON:-python}
cd "$(dirname "$0")"
until [ -d results/z/cebab_knob_cebab_ls1 ]; do sleep 60; done
# доделки: D2 newton seed 1 на phi0.1 (cusolver), разбиения эмоций st1 (последний прогон) и st2 (целиком)
$PY d2_cebab_interventions.py 1 --train-dict ngram_phi0.1 --k 1000 --probe newton --runs bce:1
$PY collapse_check.py --datasets goemotions --train-dict emotion_st1A --eval-dict emotion_st1B --k 0 --seeds 1 \
  --runs unif:1 --out-tag _split_st1 --save-z --no-raw
$PY collapse_check.py --datasets goemotions --train-dict emotion_st2A --eval-dict emotion_st2B --k 0 --seeds 0 1 \
  --runs task:0 bce:1 bce_bal:1 rand:1 recon:3 unif:1 --out-tag _split_st2 --save-z --no-raw
for s in 0 1 2; do
  $PY d2_cebab_interventions.py $s --train-dict ngram_noaspect --k 1000 --probe massmean \
    --runs task:0 bce:1 bce_bal:1 rank:1 recon:3 unif:1 vicreg:1
  $PY d2_cebab_interventions.py $s --train-dict ngram_noaspect --k 1000 --probe massmean --random-concepts --runs bce:1
  for td in ngram_phi0.1 ngram_phi0.2 ngram; do
    $PY d2_cebab_interventions.py $s --train-dict $td --k 1000 --probe massmean --runs bce:1
    $PY d2_cebab_interventions.py $s --train-dict $td --k 1000 --probe massmean --random-concepts --runs bce:1
  done
done
echo REV_D2MM_DONE
