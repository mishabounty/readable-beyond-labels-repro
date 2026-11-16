#!/bin/bash
# Доработка: симметричная сетка lambda (и доза-ответ GoEmotions без фильтра) после переполнения диска:
# каждый вызов collapse_check сразу пробуется ridge и его z удаляются (probe_and_clean.sh).
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=${PYTHON:-python}
cd "$(dirname "$0")"
GRID_OLD="0.1 0.3 3 10"; GRID_NEW="0.1 0.3 1 3 10"
runs_for() {
  local r=""
  for m in bce rank cbm cem rand unif; do for l in $GRID_OLD; do r="$r $m:$l"; done; done
  for l in 0.1 0.3 1 10; do r="$r recon:$l"; done
  for m in bce_bal vicreg; do for l in $GRID_NEW; do r="$r $m:$l"; done; done
  echo $r
}
remaining() {  # $1 -- jsonl уже сделанных прогонов, $2 -- seed; печатает прогоны сетки, которых нет
  $PY - "$1" "$2" $(runs_for) <<'PYEOF'
import json, sys
from pathlib import Path
f, seed, runs = Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3:]
done = {(r["method"], float(r["lambda"])) for r in map(json.loads, open(f)) if r["seed"] == seed} if f.exists() else set()
print(" ".join(r for r in runs if (r.split(":")[0], float(r.split(":")[1])) not in done))
PYEOF
}
# доза-ответ GoEmotions без фильтра (упала на сохранении z)
if [ "$(wc -l < results/probe_ridge_goemotions_dose_none.jsonl 2>/dev/null)" != 9 ]; then
  $PY collapse_check.py --datasets goemotions --train-dict ngram --eval-dict emotion --k 1000 --seeds 0 1 2 \
    --runs bce:1 bce_bal:1 rand:1 --out-tag _dose_none --save-z --no-raw
  ./probe_and_clean.sh goemotions_dose_none
fi
# сетка lambda
for seed in 0 1; do
  R=$(remaining results/collapse_cebab_grid.jsonl $seed)
  [ -n "$R" ] && $PY collapse_check.py --datasets cebab --seeds $seed --runs $R --out-tag _grid --save-z --no-raw --no-probe
  ./probe_and_clean.sh cebab_grid
done
for ds in cub4 cub cub4sem; do
  for seed in 0 1; do
    R=$(remaining results/collapse_${ds}_AB_grid.jsonl $seed)
    [ -n "$R" ] && $PY collapse_check.py --datasets $ds --train-dict cubA --eval-dict cubB --k 0 --emb inception \
      --seeds $seed --runs $R --out-tag _AB_grid --save-z --no-raw --no-probe
    ./probe_and_clean.sh ${ds}_AB_grid
  done
done
R=$(remaining results/collapse_goemotions_humanAB_grid.jsonl 0)
for r in $R; do  # по одному прогону: z GoEmotions по 77 МБ
  $PY collapse_check.py --datasets goemotions --train-dict emotionA --eval-dict emotionB --k 0 --seeds 0 --runs $r \
    --out-tag _humanAB_grid --save-z --no-raw --no-probe
  ./probe_and_clean.sh goemotions_humanAB_grid
done
echo REV_GRID_DONE
