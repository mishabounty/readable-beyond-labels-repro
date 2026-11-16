#!/bin/bash
#SBATCH --job-name=cem_post
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err
# Итог ночи 28.10: проба по новым моделям (пустышка-2 разбиения 0–4, Hybrid 3–4, «без понятий» 4)
# и анализ по плану HANDOFF §14.1 -> results/analysis_final.txt
cd ~/cem_repro
source venv/bin/activate
export OMP_NUM_THREADS=4
python probe_heldout.py --folds 0 1 2 3 4 --models placebo2
python probe_heldout.py --folds 3 4 --models hybrid
python probe_heldout.py --folds 4 --models noconcept
python analyze_cem.py > results/analysis_final.txt 2>&1
echo POST_DONE
