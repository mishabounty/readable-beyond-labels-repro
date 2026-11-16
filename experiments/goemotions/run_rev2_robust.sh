#!/bin/bash
# Проверки пробы (MLP, свежий сплит, [x, z], иерархический бутстрап) по мере готовности наборов.
cd "$(dirname "$0")"
PY=${PYTHON:-python}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=3
until grep -q PHASE1_GOEMO_DONE results/rev2_phase1_log.txt 2>/dev/null; do sleep 60; done
$PY conditional_probe.py finetune_goemotions_length_v2
$PY robust_probe.py finetune_goemotions_length_v2
until grep -q PHASE1_DONE results/rev2_phase1_log.txt 2>/dev/null; do sleep 60; done
$PY conditional_probe.py finetune_reuters_length_v2
$PY robust_probe.py finetune_reuters_length_v2
until grep -q COUNTRY_DONE results/rev2_phase2_log.txt 2>/dev/null; do sleep 60; done
$PY robust_probe.py finetune_reuters_country_v2 finetune_sun4_v2
echo ROBUST_DONE
