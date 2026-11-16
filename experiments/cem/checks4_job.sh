#!/bin/bash
#SBATCH --job-name=checks4
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=10
#SBATCH --mem=80G
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err
# Опыты Г1 (hybrid448 — Hybrid как в тексте статьи) и Г2 (placebo4 — пустышка «случайный код») по плану
# experiments/PLAN_2026-11-10_checks.md: 10 моделей на 4 срезах, затем проба и сводка.
cd ~/cem_repro
source venv/bin/activate
export OMP_NUM_THREADS=2
python make_placebo4.py > logs/make_placebo4.out 2>&1; echo "$(date +%H:%M) make_placebo4 exit $?"
one(){ dev=$1; m=$2; f=$3; CUDA_VISIBLE_DEVICES=$dev python run_subsample_2022.py --model $m --fold $f --num-workers 2 > logs/lane_${m}_$f.out 2> logs/lane_${m}_$f.err; echo "$(date +%H:%M) $m fold $f exit $?"; }
(one 0 hybrid448 0; one 0 hybrid448 1; one 0 placebo4 4) &
(one 1 hybrid448 2; one 1 hybrid448 3) &
(one 2 placebo4 0; one 2 placebo4 1) &
(one 3 placebo4 2; one 3 placebo4 3; one 3 hybrid448 4) &
wait
CUDA_VISIBLE_DEVICES=0 python probe_heldout.py --models hybrid448 placebo4 > logs/probe_g.out 2> logs/probe_g.err; echo "$(date +%H:%M) проба exit $?"
python analyze_2026_11_10.py > logs/analysis_2026_11_10.out 2> logs/analysis_2026_11_10.err; echo "$(date +%H:%M) сводка exit $?"
echo CHECKS4_DONE
