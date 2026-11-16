#!/bin/bash
# Один прогон HiCEM: $1 — метка конфига (R или C1), $2 — номер среза MIG внутри задания 2011.
. ~/hicem_repro/venv/bin/activate
cd ~/hicem_repro/code/experiments
export CUDA_VISIBLE_DEVICES=$2 OMP_NUM_THREADS=3
python run_experiment.py -c ~/hicem_repro/configs/cub_$1.yaml -r 1
echo "EXIT $?"
