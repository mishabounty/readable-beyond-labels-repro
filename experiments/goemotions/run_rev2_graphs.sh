#!/bin/bash
# Графы без GFT (рецензия v2, M8): GraphSAGE на Cora и PubMed, задачи topic и degree, 3 сида.
cd "$(dirname "$0")"
PY=${PYTHON:-python}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=3
$PY graph_train.py cora --task topic
$PY graph_train.py cora --task degree
until [ -f data/graph_pubmed/meta.json ]; do sleep 30; done
$PY graph_train.py pubmed --task topic
$PY graph_train.py pubmed --task degree
echo GRAPHS_DONE
