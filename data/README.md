# Data and model provenance

Large data and pretrained weights are not bundled in this repository.

## CUB-200-2011 / CEM

Use the official CUB-200-2011 release and the `class_attr_data_10` files distributed with the CEM code. The expected
directory layout is the one used by `cem.data.CUB200.cub_loader`. The upstream CEM source is recorded in
`../upstream/cem/` and its README/license.

## HiCEM

Use the CUB split files expected by `upstream/hicem/cemcd/data/cub.py`. The original HiCEM repository also supports
MNIST-ADD, SHAPES, AwA2, PseudoKitchens, and ImageNet; this paper's reported check uses the CUB branch only.

## DreamSim

Download the NIGHTS, TID2013, and KADID-10k evaluation data from their original sources and obtain the DINO/DreamSim
weights through the upstream DreamSim setup. The expected model and dataset paths are documented in
`upstream/dreamsim/README.md` and the paper-specific `experiments/dreamsim/*.sh` scripts.

## GoEmotions

Use the public GoEmotions release and generate embeddings with the model specified in the text experiment notes. Do
not commit raw user text, downloaded model weights, embedding caches, or access tokens.

## Reproducibility note

The exact compact outputs used for the manuscript are already in `../results/`. A fresh run can therefore be compared
against the recorded JSON/JSONL files without distributing the underlying copyrighted datasets or large checkpoints.
