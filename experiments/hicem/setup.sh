#!/bin/bash
# Установка кода HiCEM (Hill et al., ICLR 2026) для быстрой проверки: коммит 4b665fa7a4 (9.12.2025),
# последняя версия, где для CUB используется поиск подпонятий SAE, как в статье.
set -e
cd ~/hicem_repro
[ -d code ] || git clone https://github.com/OscarPi/cem-concept-discovery.git code
cd code && git checkout -q 4b665fa7a4 && git rev-parse HEAD > ../COMMIT.txt
python3 -m pip install --user --quiet uv
~/.local/bin/uv python install 3.11
[ -d ~/hicem_repro/venv ] || ~/.local/bin/uv venv --python 3.11 ~/hicem_repro/venv
. ~/hicem_repro/venv/bin/activate
~/.local/bin/uv pip install -e .
~/.local/bin/uv pip install git+https://github.com/openai/CLIP.git
python -c "import torch, clip, lightning; print(\"torch\", torch.__version__, \"cuda build\", torch.version.cuda)"
mkdir -p ~/hicem_repro/datasets/CUB ~/hicem_repro/checkpoints ~/hicem_repro/results
ln -sfn ~/cem_repro/data/CUB_200_2011/images ~/hicem_repro/datasets/CUB/images
ln -sfn ~/cem_repro/data/CUB_200_2011/attributes ~/hicem_repro/datasets/CUB/attributes
cp -r ~/hicem_repro/code/splits/CUB/class_attr_data_10 ~/hicem_repro/datasets/CUB/
echo SETUP_DONE
