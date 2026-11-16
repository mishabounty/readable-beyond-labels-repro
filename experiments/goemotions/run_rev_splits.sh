#!/bin/bash
# Доработка: несколько разбиений эмоций GoEmotions (5 случайных + 3 стратифицированных по тональности),
# затем доза-ответ по порогу утечки phi для n-граммных словарей (CEBaB, GoEmotions).
PY=${PYTHON:-python}
cd "$(dirname "$0")"
for sp in r1 r2 r3 r4 r5 st0 st1 st2; do
  $PY collapse_check.py --datasets goemotions --train-dict emotion_${sp}A --eval-dict emotion_${sp}B --k 0 \
    --seeds 0 1 --runs task:0 bce:1 bce_bal:1 rand:1 recon:3 unif:1 --out-tag _split_$sp --save-z --no-raw
done
for ds in cebab goemotions; do
  ev=$([ $ds = cebab ] && echo aspect || echo emotion)
  for lv in phi0.05 phi0.1 phi0.2 none; do
    td=$([ $lv = none ] && echo ngram || echo ngram_$lv)
    $PY collapse_check.py --datasets $ds --train-dict $td --eval-dict $ev --k 1000 \
      --seeds 0 1 2 --runs bce:1 bce_bal:1 rand:1 --out-tag _dose_$lv --save-z --no-raw
  done
done
echo REV_SPLITS_DONE
