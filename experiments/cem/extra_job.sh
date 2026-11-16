#!/bin/bash
#SBATCH --job-name=cem_extra
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=12
#SBATCH --mem=112G
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err
# Дополнительные проверки 29.10 (4 потока на 4 срезах MIG):
#  0: признаки CEM / пустышек (extract_features.py) → «CEM без понятий» разбиения 0, 2 → HiCEM R, ещё прогон;
#  1: признаки Hybrid / «без понятий» / исходной ResNet (ImageNet) → «CEM без понятий» 1, 3, 4;
#  2: HiCEM C1, ещё прогон;  3: HiCEM C1 ещё прогон (через 3 мин, чтобы не столкнуться с именем папки) → HiCEM R.
# Старый venv (torch 1.12) видит срезы по номеру, venv HiCEM (torch 2.14) — только по UUID.
cd ~/cem_repro
U0=MIG-b290b314-fc27-5cfb-af39-2b8b8657e3c6; U2=MIG-96dcb4fd-451c-5ea5-9fc6-895370b4c4b3; U3=MIG-d4bbf7bd-e735-54e4-8af0-f64f9a0afded
old(){ (source ~/cem_repro/venv/bin/activate; CUDA_VISIBLE_DEVICES=$1 OMP_NUM_THREADS=3 "${@:2}"); }
hicem(){ (source ~/hicem_repro/venv/bin/activate; cd ~/hicem_repro/code/experiments; CUDA_VISIBLE_DEVICES=$1 OMP_NUM_THREADS=3 python run_experiment.py -c ~/hicem_repro/configs/cub_$2.yaml -r 1); }
cemnoc(){ old $1 python run_subsample_2022.py --model cemnoc --fold $2 --num-workers 3 > logs/lane_cemnoc_$2.out 2> logs/lane_cemnoc_$2.err; echo "$(date +%H:%M) cemnoc fold $2 exit $?"; old $1 python extract_features.py --models cemnoc --folds $2 >> logs/feat_cemnoc.out 2>> logs/feat_cemnoc.err; echo "$(date +%H:%M) cemnoc features fold $2 exit $?"; }
lane0(){ old 0 python extract_features.py --models cem placebo placebo2 > logs/feat_a.out 2> logs/feat_a.err; echo "$(date +%H:%M) features A exit $?"; cemnoc 0 0; cemnoc 0 2; hicem $U0 R > ~/hicem_repro/logs/R_s2.log 2>&1; echo "$(date +%H:%M) hicem R exit $?"; }
lane1(){ old 1 python extract_features.py --models hybrid noconcept --pretrained > logs/feat_b.out 2> logs/feat_b.err; echo "$(date +%H:%M) features B exit $?"; cemnoc 1 1; cemnoc 1 3; cemnoc 1 4; }
lane2(){ hicem $U2 C1 > ~/hicem_repro/logs/C1_s2.log 2>&1; echo "$(date +%H:%M) hicem C1 exit $?"; }
lane3(){ sleep 180; hicem $U3 C1 > ~/hicem_repro/logs/C1_s3.log 2>&1; echo "$(date +%H:%M) hicem C1 exit $?"; hicem $U3 R > ~/hicem_repro/logs/R_s3.log 2>&1; echo "$(date +%H:%M) hicem R exit $?"; }
lane0 & lane1 & lane2 & lane3 & wait; echo ALL_DONE
