#!/bin/bash
#SBATCH --job-name=checks3
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=48G
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err
# Опыты Э2, Э4, Э5 (план experiments/PLAN_2026-11-04_checks.md): сначала выходы голов на GPU и пробный прогон, затем полный счёт.
set -e
cd ~/cem_repro
source venv/bin/activate
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1
python checks3.py head
SMOKE=1 NJ=8 python checks3.py e2 e5
NJ=8 python checks3.py e2 e4 e5
