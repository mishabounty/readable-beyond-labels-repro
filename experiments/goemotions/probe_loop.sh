#!/bin/bash
# Пробы по сохранённым z, пока идут обучения (повторные проходы подхватывают новые z):
#   ridge (probe_ridge.py) -- все наборы; логистическая регрессия до сходимости (probe_eval.py) -- основные постановки.
PY=${PYTHON:-python}
cd "$(dirname "$0")"
LR_TEXT="goemotions_humanAB_v2 goemotions_v2 finetune_goemotions_v2 goemotions_humanAB_tfidf"
LR_CUB="cub4_AB_v2 finetune_cub4_v2 cub_AB_v2 cub4sem_AB_v2"
pass() {
  for d in results/z/*/; do
    n=$(basename $d)
    case $n in *_v2|*_tfidf) extra=;; *) extra=--no-white;; esac
    $PY probe_ridge.py $n $extra
  done
  for n in $LR_TEXT; do $PY probe_eval.py $n; done
  for n in $LR_CUB; do $PY probe_eval.py $n --white-input-only; done
}
while true; do
  pass
  if grep -q REV_ALL_DONE results/rev_marker.txt 2>/dev/null; then pass; echo PROBE_DONE; break; fi
  sleep 120
done
