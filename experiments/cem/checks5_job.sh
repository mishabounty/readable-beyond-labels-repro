#!/bin/bash
#SBATCH --job-name=checks5
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=12
#SBATCH --mem=80G
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err
# План experiments/PLAN_2026-11-11_amendments.md: П2 (placebo4w, 5 повторов), признаки placebo4 / placebo4w / hybrid448,
# проба, сводка П2 и П4, затем П3 (избирательность, CPU). Коды выхода сохраняются до date.
cd ~/cem_repro
source venv/bin/activate
export OMP_NUM_THREADS=2
one(){ dev=$1; f=$2; CUDA_VISIBLE_DEVICES=$dev python run_subsample_2022.py --model placebo4w --fold $f --num-workers 2 > logs/lane_placebo4w_$f.out 2> logs/lane_placebo4w_$f.err; e=$?; echo "$(date +%H:%M) placebo4w fold $f exit $e"; CUDA_VISIBLE_DEVICES=$dev python extract_features.py --models placebo4w --folds $f >> logs/feat_p4w.out 2>> logs/feat_p4w.err; e=$?; echo "$(date +%H:%M) признаки placebo4w $f exit $e"; }
(one 0 0; one 0 4) &
(one 1 1; CUDA_VISIBLE_DEVICES=1 python extract_features.py --models placebo4 --folds 0 1 2 3 4 >> logs/feat_p4.out 2>> logs/feat_p4.err; e=$?; echo "$(date +%H:%M) признаки placebo4 exit $e") &
(one 2 2; CUDA_VISIBLE_DEVICES=2 python extract_features.py --models hybrid448 --folds 0 1 2 3 4 >> logs/feat_h448.out 2>> logs/feat_h448.err; e=$?; echo "$(date +%H:%M) признаки hybrid448 exit $e") &
(one 3 3) &
wait
CUDA_VISIBLE_DEVICES=0 python probe_heldout.py --models placebo4w > logs/probe_p4w.out 2> logs/probe_p4w.err; e=$?; echo "$(date +%H:%M) проба exit $e"
python analyze_2026_11_11.py > logs/analysis_2026_11_11.out 2> logs/analysis_2026_11_11.err; e=$?; echo "$(date +%H:%M) сводка П2 exit $e"
CUDA_VISIBLE_DEVICES= python analyze_selectivity.py --models cem placebo placebo2 placebo3 placebo4 placebo4w cemnoc hybrid hybrid448 > logs/analysis_selectivity.out 2> logs/analysis_selectivity.err; e=$?; echo "$(date +%H:%M) избирательность exit $e"
echo CHECKS5_DONE
