#!/bin/bash
#SBATCH --job-name=dreamsim2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=12
#SBATCH --mem=80G
#SBATCH --output=${REPRO_ROOT:-.}/logs/%x.%j.out
#SBATCH --error=${REPRO_ROOT:-.}/logs/%x.%j.err
# План experiments/PLAN_2026-11-11_amendments.md, П1: пустышка «случайная проекция» ×5, оценка, сводка по новому правилу.
H=~/dreamsim_repro; cd $H/code
source $H/venv/bin/activate
export WANDB_MODE=offline WANDB_SILENT=true OMP_NUM_THREADS=2 PYTHONPATH=$H/tf32:$H/code  # TF32 как в 2052
U=(MIG-b290b314-fc27-5cfb-af39-2b8b8657e3c6 MIG-adff96b3-16de-518a-9559-681cc708eb18 MIG-96dcb4fd-451c-5ea5-9fc6-895370b4c4b3 MIG-d4bbf7bd-e735-54e4-8af0-f64f9a0afded)
CUDA_VISIBLE_DEVICES=${U[0]} python $H/make_randproj.py > $H/logs/randproj.out 2>&1; e=$?; echo "$(date +%H:%M) случайная проекция exit $e"; [ $e -eq 0 ] || exit 1
run(){ dev=$1; name=$2; root=$3; seed=$4; CUDA_VISIBLE_DEVICES=$dev python training/train.py --config configs/train_single_model_lora.yaml --dataset_root $H/dataset/$root --seed $seed --tag $name --log_dir $H/runs/$name --load_dir $H/models --num_workers 2 > $H/logs/train_$name.out 2>&1; e=$?; echo "$(date +%H:%M) $name exit $e"; }
(run ${U[0]} rp_s0 nights_rp_s0 100; run ${U[0]} rp_s4 nights_rp_s4 104) &
(run ${U[1]} rp_s1 nights_rp_s1 101) &
(run ${U[2]} rp_s2 nights_rp_s2 102) &
(run ${U[3]} rp_s3 nights_rp_s3 103) &
wait
ev(){ dev=$1; shift; CUDA_VISIBLE_DEVICES=$dev python $H/eval_b.py --models "$@" >> $H/logs/eval2_$dev.out 2>&1; e=$?; echo "$(date +%H:%M) оценка $* exit $e"; }
ev ${U[0]} rp_s0 rp_s4 & ev ${U[1]} rp_s1 & ev ${U[2]} rp_s2 & ev ${U[3]} rp_s3 &
wait
python $H/summarize.py > $H/logs/summary2.out 2>&1; e=$?; echo "$(date +%H:%M) сводка exit $e"
echo DREAMSIM2_DONE
