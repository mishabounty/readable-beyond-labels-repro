#!/bin/bash
#SBATCH --job-name=checks2_cpu
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=6
#SBATCH --mem=40G
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err
# Проверки по рецензиям на v8 без GPU (план experiments/PLAN_2026-11-02_checks.md): N1/N4/N5 на моделях, видевших все
# виды (analyze_nested.py), N6 (mlp_tuned.py), N7 (hicem_checks2.py, K = 1000).
cd ~/cem_repro
source ~/cem_repro/venv/bin/activate
NJ=6 python analyze_nested.py > logs/analysis_nested.out 2> logs/analysis_nested.err; echo "$(date +%H:%M) nested exit $?"
NJ=6 python mlp_tuned.py > logs/mlp_tuned.out 2> logs/mlp_tuned.err; echo "$(date +%H:%M) mlp exit $?"
deactivate
(source ~/hicem_repro/venv/bin/activate; cd ~/hicem_repro/code/experiments; CUDA_VISIBLE_DEVICES= K=1000 python hicem_checks2.py ~/hicem_repro/configs/cub_R.yaml > ~/hicem_repro/logs/hicem_checks2.out 2> ~/hicem_repro/logs/hicem_checks2.err); echo "$(date +%H:%M) hicem2 exit $?"
echo CPU2_DONE
