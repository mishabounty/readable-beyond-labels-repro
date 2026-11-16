#!/bin/bash
# Проверка опыта CEM §5.3, 5 запусков: CEM (lambda 0.1) на настоящих атрибутах и на пустышке, CEM без понятий,
# только задача (полностью и 2 эпохи -- та же точность по видам, что у CEM), Hybrid-CBM; затем чтение
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=3
PY=${PYTHON:-python}
$PY cem_check.py --tag _full --seeds 0 1 2 3 4 --epochs 60 --patience 10 --head-lr 1e-3 \
  --runs task:0:cemA task:0:cemA:ep2 cem:0.1:cemA cem:0.1:cemArnd cem:0:cemA hcbm:0.1:cemA
echo "TRAIN DONE"
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 $PY cem_readout.py --tag _full 2>&1 | grep -v -i warning > results/cem_readout_full.log
echo "READOUT DONE"
