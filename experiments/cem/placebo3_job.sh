#!/bin/bash
#SBATCH --job-name=placebo3
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=12
#SBATCH --mem=80G
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err
# Пустышка CEM, зависящая от входа (план experiments/PLAN_2026-11-04_placebo3.md): 5 повторов на 4 срезах, признаки, проба, анализ.
cd ~/cem_repro
source venv/bin/activate
export OMP_NUM_THREADS=2
one(){ dev=$1; f=$2; CUDA_VISIBLE_DEVICES=$dev python run_subsample_2022.py --model placebo3 --fold $f --num-workers 2 > logs/lane_placebo3_$f.out 2> logs/lane_placebo3_$f.err; echo "$(date +%H:%M) placebo3 fold $f exit $?"; CUDA_VISIBLE_DEVICES=$dev python extract_features.py --models placebo3 --folds $f >> logs/feat_placebo3.out 2>> logs/feat_placebo3.err; echo "$(date +%H:%M) признаки fold $f exit $?"; }
(one 0 0; one 0 4) & one 1 1 & one 2 2 & one 3 3 & wait
CUDA_VISIBLE_DEVICES=0 python probe_heldout.py --models placebo3 cemnoc > logs/probe_placebo3.out 2> logs/probe_placebo3.err; echo "$(date +%H:%M) проба exit $?"
python analyze_placebo3.py > logs/analysis_placebo3.out 2> logs/analysis_placebo3.err; echo "$(date +%H:%M) сводка exit $?"
OMP_NUM_THREADS=1 NJ=11 python analyze_nested.py --models cem placebo3 placebo2 cemnoc pretrained --pairs cem:placebo3 placebo3:placebo2 placebo3:cemnoc cem:placebo2 --tag _p3 > logs/analysis_nested_p3.out 2> logs/analysis_nested_p3.err; echo "$(date +%H:%M) вложенная проба exit $?"
echo PLACEBO3_DONE
