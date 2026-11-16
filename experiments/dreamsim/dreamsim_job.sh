#!/bin/bash
#SBATCH --job-name=dreamsim
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=12
#SBATCH --mem=80G
#SBATCH --output=${REPRO_ROOT:-.}/logs/%x.%j.out
#SBATCH --error=${REPRO_ROOT:-.}/logs/%x.%j.err
# DreamSim с пустышками (план experiments/PLAN_2026-11-10_dreamsim.md): ждёт данные, пробный прогон, 13 обучений на 4
# срезах (human ×5, shuf ×5, anti ×3), оценка (NIGHTS test, TID2013, KADID-10k), сводка.
# venv на torch 2.x видит срезы MIG только по UUID.
H=~/dreamsim_repro; cd $H/code
source $H/venv/bin/activate
export WANDB_MODE=offline WANDB_SILENT=true OMP_NUM_THREADS=2 PYTHONPATH=$H/tf32:$H/code  # tf32/sitecustomize.py включает TF32 (одинаково для всех вариантов)
U=(MIG-b290b314-fc27-5cfb-af39-2b8b8657e3c6 MIG-adff96b3-16de-518a-9559-681cc708eb18 MIG-96dcb4fd-451c-5ea5-9fc6-895370b4c4b3 MIG-d4bbf7bd-e735-54e4-8af0-f64f9a0afded)
python -c "import wandb" 2>/dev/null || python -m pip install -q wandb
until grep -q DL3_DONE $H/logs/dl_nights3.log; do sleep 60; done
n=$(ls -a $H/dataset/nights | grep -c "^\.done_"); echo "$(date +%H:%M) частей NIGHTS: $n из 200"; [ "$n" -eq 200 ] || { echo "NIGHTS неполный — стоп"; exit 1; }
python $H/make_label_variants.py > $H/logs/label_variants.out 2>&1; echo "$(date +%H:%M) варианты ответов exit $?"
python -c "import torch; torch.hub.set_dir('$H/models'); torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')" > $H/logs/hub.out 2>&1; echo "$(date +%H:%M) DINO в кэше exit $?"
# пробный прогон: 256 троек, 1 эпоха, затем оценка на 64 парах
python - <<PY
import pandas as pd, os
s = pd.read_csv("$H/dataset/nights/data.csv"); s = s[s.votes >= 6]
t = pd.concat([s[s.split == "train"].head(256), s[s.split == "val"].head(64), s[s.split == "test"].head(64)])
d = "$H/dataset/nights_smoke"; os.makedirs(d, exist_ok=True)
for sub in ("ref", "distort"):
    os.path.islink(f"{d}/{sub}") or os.symlink(f"$H/dataset/nights/{sub}", f"{d}/{sub}")
t.to_csv(f"{d}/data.csv", index=False)
PY
CUDA_VISIBLE_DEVICES=${U[0]} python training/train.py --config configs/train_single_model_lora.yaml --dataset_root $H/dataset/nights_smoke --seed 1 --tag smoke --log_dir $H/runs/smoke --load_dir $H/models --num_workers 2 --epochs 1 > $H/logs/train_smoke.out 2>&1; echo "$(date +%H:%M) пробное обучение exit $?"
ls -R $H/runs/smoke | grep -q "epoch_0" || { echo "пробное обучение не сохранило адаптер — стоп"; exit 1; }
CUDA_VISIBLE_DEVICES=${U[0]} LIMIT=64 NIGHTS_ROOT=nights_smoke RES_DIR=results_smoke python $H/eval_b.py --models base authors smoke > $H/logs/eval_smoke.out 2>&1; e=$?; echo "$(date +%H:%M) пробная оценка exit $e"; [ $e -eq 0 ] || { echo "пробная оценка упала — стоп"; exit 1; }
run(){ dev=$1; name=$2; root=$3; seed=$4; CUDA_VISIBLE_DEVICES=$dev python training/train.py --config configs/train_single_model_lora.yaml --dataset_root $H/dataset/$root --seed $seed --tag $name --log_dir $H/runs/$name --load_dir $H/models --num_workers 2 > $H/logs/train_$name.out 2>&1; echo "$(date +%H:%M) $name exit $?"; }
(run ${U[0]} human_s0 nights 100; run ${U[0]} shuf_s0 nights_shuf_s0 100; run ${U[0]} anti_s0 nights_anti 100; run ${U[0]} human_s4 nights 104) &
(run ${U[1]} human_s1 nights 101; run ${U[1]} shuf_s1 nights_shuf_s1 101; run ${U[1]} anti_s1 nights_anti 101) &
(run ${U[2]} human_s2 nights 102; run ${U[2]} shuf_s2 nights_shuf_s2 102; run ${U[2]} anti_s2 nights_anti 102) &
(run ${U[3]} human_s3 nights 103; run ${U[3]} shuf_s3 nights_shuf_s3 103; run ${U[3]} shuf_s4 nights_shuf_s4 104) &
wait
ev(){ dev=$1; shift; CUDA_VISIBLE_DEVICES=$dev python $H/eval_b.py --models "$@" >> $H/logs/eval_$dev.out 2>&1; echo "$(date +%H:%M) оценка $* exit $?"; }
ev ${U[0]} base authors human_s0 shuf_s0 &
ev ${U[1]} human_s1 shuf_s1 anti_s0 human_s4 &
ev ${U[2]} human_s2 shuf_s2 anti_s1 &
ev ${U[3]} human_s3 shuf_s3 anti_s2 shuf_s4 &
wait
python $H/summarize.py > $H/logs/summary.out 2>&1; echo "$(date +%H:%M) сводка exit $?"
echo DREAMSIM_DONE
