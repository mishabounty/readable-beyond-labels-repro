#!/bin/bash
# Пробный прогон DreamSim до основного задания: 64/16/16 троек, 1 эпоха, batch 8; оценка на 16 парах каждого набора.
H=~/dreamsim_repro; cd $H/code; source $H/venv/bin/activate
export WANDB_MODE=offline WANDB_SILENT=true OMP_NUM_THREADS=2 PYTHONPATH=$H/code
python - <<PY
import pandas as pd, os
H = os.path.expanduser("~/dreamsim_repro")
s = pd.read_csv(f"{H}/dataset/nights/data.csv"); s = s[s.votes >= 6]
t = pd.concat([s[s.split == "train"].head(64), s[s.split == "val"].head(16), s[s.split == "test"].head(16)])
d = f"{H}/dataset/nights_smoke_pre"; os.makedirs(d, exist_ok=True)
for sub in ("ref", "distort"):
    os.path.islink(f"{d}/{sub}") or os.symlink(f"{H}/dataset/nights/{sub}", f"{d}/{sub}")
t.to_csv(f"{d}/data.csv", index=False)
print("троек:", dict(t.split.value_counts()))
PY
rm -rf $H/runs/smoke_pre $H/results_smoke_pre
python training/train.py --config configs/train_single_model_lora.yaml --dataset_root $H/dataset/nights_smoke_pre --seed 1 --tag smoke_pre --log_dir $H/runs/smoke_pre --load_dir $H/models --num_workers 2 --epochs 1 --batch_size 8; echo "train exit $?"
find $H/runs/smoke_pre -name adapter_config.json | head -3
LIMIT=16 NIGHTS_ROOT=nights_smoke_pre RES_DIR=results_smoke_pre python $H/eval_b.py --models base authors smoke_pre; echo "eval exit $?"
