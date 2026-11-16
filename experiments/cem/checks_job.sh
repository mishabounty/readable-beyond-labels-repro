#!/bin/bash
#SBATCH --job-name=checks_gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err
# Проверки по рецензиям 30.10 на GPU (план experiments/PLAN_2026-10-30_checks.md, пункты C и D):
#  потоки 0–2: CEM / пустышка-2 / CEM без понятий, обученные только на видах S1 (make_half_species.py), 5 повторов,
#              признаки после каждого повтора; затем по прогону HiCEM (Rnull зёрна банка 1 и 2; Rtrue);
#  поток 3: HiCEM P (зёрна пустышки 3001, 3002) и Rtrue.
# Старый venv (torch 1.12) видит срезы по номеру, venv HiCEM (torch 2.14) — только по UUID.
cd ~/cem_repro
U0=MIG-b290b314-fc27-5cfb-af39-2b8b8657e3c6; U1=MIG-adff96b3-16de-518a-9559-681cc708eb18; U2=MIG-96dcb4fd-451c-5ea5-9fc6-895370b4c4b3; U3=MIG-d4bbf7bd-e735-54e4-8af0-f64f9a0afded
old(){ (source ~/cem_repro/venv/bin/activate; CUDA_VISIBLE_DEVICES=$1 OMP_NUM_THREADS=2 "${@:2}"); }
ctrl(){ (source ~/hicem_repro/venv/bin/activate; cd ~/hicem_repro/code/experiments; CUDA_VISIBLE_DEVICES=$1 OMP_NUM_THREADS=2 python run_ctrl.py ~/hicem_repro/configs/cub_$2.yaml $3 $4); }
half(){ dev=$1; m=$2; for f in 0 1 2 3 4; do old $dev python run_subsample_2022.py --model $m --fold $f --num-workers 2 > logs/lane_${m}_$f.out 2> logs/lane_${m}_$f.err; echo "$(date +%H:%M) $m fold $f exit $?"; old $dev python extract_features.py --models $m --folds $f >> logs/feat_half.out 2>> logs/feat_half.err; echo "$(date +%H:%M) $m features fold $f exit $?"; done; }
FLAG=logs/half_data.ready; rm -f $FLAG
lane0(){ old 0 python make_half_species.py > logs/make_half.out 2>&1; echo "$(date +%H:%M) make_half exit $?"; touch $FLAG; half 0 cem_half; ctrl $U0 Rnull null_bank 1 > ~/hicem_repro/logs/Rnull_s1.log 2>&1; echo "$(date +%H:%M) hicem Rnull 1 exit $?"; }
lane1(){ while [ ! -f $FLAG ]; do sleep 10; done; half 1 placebo2_half; sleep 240; ctrl $U1 Rnull null_bank 2 > ~/hicem_repro/logs/Rnull_s2.log 2>&1; echo "$(date +%H:%M) hicem Rnull 2 exit $?"; }
lane2(){ while [ ! -f $FLAG ]; do sleep 10; done; half 2 cemnoc_half; ctrl $U2 Rtrue true_filter > ~/hicem_repro/logs/Rtrue_s2.log 2>&1; echo "$(date +%H:%M) hicem Rtrue exit $?"; }
lane3(){ ctrl $U3 P placebo 3001 > ~/hicem_repro/logs/P_s3001.log 2>&1; echo "$(date +%H:%M) hicem P 3001 exit $?"; ctrl $U3 P placebo 3002 > ~/hicem_repro/logs/P_s3002.log 2>&1; echo "$(date +%H:%M) hicem P 3002 exit $?"; sleep 300; ctrl $U3 Rtrue true_filter > ~/hicem_repro/logs/Rtrue_s3.log 2>&1; echo "$(date +%H:%M) hicem Rtrue exit $?"; }
lane0 & lane1 & lane2 & lane3 & wait; echo ALL_DONE
