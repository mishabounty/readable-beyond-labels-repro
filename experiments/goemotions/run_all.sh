#!/usr/bin/env bash
# Полная сетка: словари n-грамм разного размера, внешние концепты, контроль со случайным словарём.
set -u
cd "$(dirname "$0")"
PY=${PYTHON:-python}
SEEDS="0 1 2"
for ds in cebab goemotions agnews 20ng; do
  for k in 100 1000 5000; do
    $PY train.py --dataset $ds --dict ngram --k $k --methods task bce rank --lambdas 0.3 1.0 --seeds $SEEDS --raw-reference --tag main
  done
done
$PY train.py --dataset goemotions --dict emotion --methods task bce rank --lambdas 0.3 1.0 --seeds $SEEDS --raw-reference --tag main
$PY train.py --dataset cebab --dict aspect --methods task bce rank --lambdas 0.3 1.0 --seeds $SEEDS --raw-reference --tag main
for ds in agnews goemotions; do
  $PY train.py --dataset $ds --dict ngram --k 1000 --methods task bce rank --lambdas 1.0 --seeds $SEEDS --random-concepts --tag random
done
echo ALL_DONE
