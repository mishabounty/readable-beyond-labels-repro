#!/bin/bash
#SBATCH --job-name=hicem_c1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --output=${REPRO_ROOT:-.}/logs/%x.%j.out
#SBATCH --error=${REPRO_ROOT:-.}/logs/%x.%j.err
# Страховка: если прогон C1 внутри задания 2011 оборвался до главного числа, повторить его целиком.
F=~/hicem_repro/results/C1/cub-ALPHA-ALPHA-ALPHA/results.yaml
if grep -q "enhanced_clip_vitl14_hicem_discovered_concept_auc" $F 2>/dev/null; then echo "C1 уже готов, перезапуск не нужен"; exit 0; fi
. ~/hicem_repro/venv/bin/activate
cd ~/hicem_repro/code/experiments
export OMP_NUM_THREADS=4
python run_experiment.py -c ~/hicem_repro/configs/cub_C1.yaml -r 1
echo "EXIT $?"
