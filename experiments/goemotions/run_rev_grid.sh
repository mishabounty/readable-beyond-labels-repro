#!/bin/bash
# Доработка, вторая очередь (после замороженных прогонов v2):
#  1) D2 на контрфактах CEBaB с пробой до сходимости, все цели этапа 1 + доза-ответ по словарю;
#  2) прямые ручки схлопывания при неизменной задаче (только task): logit scale, label smoothing, weight decay, длина;
#  3) энкодер с нуля на TF-IDF (GoEmotions 13 -> 14 эмоций);
#  4) симметричная сетка lambda для всех целей.
# Читаемость для 2)-4) считает probe_eval.py по сохранённым z (--no-probe).
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3
PY=${PYTHON:-python}
cd "$(dirname "$0")"
until grep -q FROZEN_DONE results/v2_frozen_log.txt; do sleep 60; done
for s in 0 1 2; do
  $PY d2_cebab_interventions.py $s --train-dict ngram_noaspect --k 1000 --probe newton \
    --runs task:0 bce:1 bce_bal:1 rank:1 recon:3 unif:1 vicreg:1
  $PY d2_cebab_interventions.py $s --train-dict ngram_noaspect --k 1000 --probe newton --random-concepts --runs bce:1
  for td in ngram_phi0.1 ngram_phi0.2 ngram; do
    $PY d2_cebab_interventions.py $s --train-dict $td --k 1000 --probe newton --runs bce:1
    $PY d2_cebab_interventions.py $s --train-dict $td --k 1000 --probe newton --random-concepts --runs bce:1
  done
done
knob() {  # $1 -- набор (датасет и словари), $2 -- имя ручки, остальное -- флаги
  local set=$1 name=$2; shift 2
  case $set in
    cebab) a="--datasets cebab";;
    goemotions_humanAB) a="--datasets goemotions --train-dict emotionA --eval-dict emotionB --k 0";;
    cub4_AB) a="--datasets cub4 --train-dict cubA --eval-dict cubB --k 0 --emb inception";;
    cub_AB) a="--datasets cub --train-dict cubA --eval-dict cubB --k 0 --emb inception";;
  esac
  $PY collapse_check.py $a --seeds 0 1 --runs task:0 --out-tag _knob_$name --save-z --no-raw --no-probe "$@"
}
for set in cebab cub4_AB cub_AB goemotions_humanAB; do
  knob $set ${set}_ls1 --logit-scale 1; knob $set ${set}_ls3 --logit-scale 3; knob $set ${set}_ls30 --logit-scale 30
  knob $set ${set}_sm0.1 --label-smoothing 0.1; knob $set ${set}_sm0.3 --label-smoothing 0.3
  knob $set ${set}_wd0 --wd 0; knob $set ${set}_wd0.1 --wd 0.1; knob $set ${set}_wd1 --wd 1
  for e in 1 3 10 40; do knob $set ${set}_ep$e --fixed-epochs $e; done
done
$PY collapse_check.py --datasets goemotions --train-dict emotionA --eval-dict emotionB --k 0 --emb tfidf --seeds 0 1 2 \
  --runs task:0 bce:1 bce_bal:1 rank:1 rand:1 recon:3 unif:1 vicreg:1 cbm:1 cem:1 --out-tag _humanAB_tfidf --save-z --no-raw --no-probe
GRID_OLD="0.1 0.3 3 10"; GRID_NEW="0.1 0.3 1 3 10"
runs_for() {
  local r=""
  for m in bce rank cbm cem rand unif; do for l in $GRID_OLD; do r="$r $m:$l"; done; done
  for l in 0.1 0.3 1 10; do r="$r recon:$l"; done
  for m in bce_bal vicreg; do for l in $GRID_NEW; do r="$r $m:$l"; done; done
  echo $r
}
RUNS=$(runs_for)
$PY collapse_check.py --datasets cebab --seeds 0 1 --runs $RUNS --out-tag _grid --save-z --no-raw --no-probe
for ds in cub4 cub cub4sem; do
  $PY collapse_check.py --datasets $ds --train-dict cubA --eval-dict cubB --k 0 --emb inception --seeds 0 1 --runs $RUNS \
    --out-tag _AB_grid --save-z --no-raw --no-probe
done
$PY collapse_check.py --datasets goemotions --train-dict emotionA --eval-dict emotionB --k 0 --seeds 0 --runs $RUNS \
  --out-tag _humanAB_grid --save-z --no-raw --no-probe
echo REV_GRID_DONE
