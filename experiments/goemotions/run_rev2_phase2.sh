#!/bin/bash
# Доработка 2, фаза 2 (после фазы 1): второй набор изображений (SUN), естественная задача (Reuters, страна),
# методы против забывания, гранулярность задачи на CUB, дополнительные сиды. Блоки независимы: ошибка одного
# не останавливает следующие.
cd "$(dirname "$0")"
PY=${PYTHON:-python}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=3
until grep -q PHASE1_DONE results/rev2_phase1_log.txt 2>/dev/null; do sleep 60; done
# 2. Reuters, естественная задача: страна (PLACES)
$PY text_finetune.py --dataset reuters --task country --max-len 128 --seeds 0 1 2 --runs task:0 rand:1 bce:1 --out-tag _v2 --save-z
$PY probe_ridge.py finetune_reuters_country_v2 --no-white
$PY conditional_probe.py finetune_reuters_country_v2
echo COUNTRY_DONE
# 3. Гранулярность задачи на CUB: случайные 8 / 16 / 50 суперклассов
for K in 8 16 50; do
  $PY cub_finetune.py --dataset cub$K --seeds 0 1 2 --runs task:0 --out-tag _v2 --save-z --no-probe
done
$PY probe_ridge.py finetune_cub8_v2 finetune_cub16_v2 finetune_cub50_v2 --no-white
$PY conditional_probe.py finetune_cub8_v2 finetune_cub16_v2 finetune_cub50_v2
echo CUBK_DONE
# 4. Методы против забывания (GoEmotions, задача «длина»)
$PY text_finetune.py --task length --seeds 0 1 2 --runs wise:0 lpft:0 l2sp:0.01 --out-tag _v2 --save-z
$PY probe_ridge.py finetune_goemotions_length_v2 --no-white
echo BASELINES_DONE
# 5. Дополнительные сиды для контролей без смысла (GoEmotions) и тот же набор контролей на Reuters
$PY text_finetune.py --task length --seeds 3 4 --runs hyp:1 recon:3 unif:1 --out-tag _v2 --save-z
$PY text_finetune.py --dataset reuters --task length --max-len 128 --seeds 0 1 2 --runs hyp:1 recon:3 unif:1 --out-tag _v2 --save-z
$PY probe_ridge.py finetune_goemotions_length_v2 finetune_reuters_length_v2 --no-white
$PY conditional_probe.py finetune_goemotions_length_v2 finetune_reuters_length_v2
echo PHASE2_DONE
