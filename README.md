# Reproducibility package

This repository contains the code and compact result files for the experiments reported in
**Readable beyond the labels? Re-examining claims about human concept supervision**.

The package is deliberately small. It includes analysis scripts and the final JSON/JSONL outputs, but it does not
redistribute CUB, NIGHTS, TID2013, KADID-10k, GoEmotions, pretrained weights, checkpoints, or the large training
logs. The data and model provenance, expected locations, and the upstream commits are recorded below.

## What can be checked immediately

The `results/` tree contains the compact outputs used for the paper's tables and checks:

* `results/cem/` — the five-run CEM, Hybrid-CBM, no-concept, and label-control outputs;
* `results/hicem/` — HiCEM/Concept Splitting checks and matched null results;
* `results/dreamsim/` — DreamSim evaluation summaries for human, shuffled, anti-human, and random-projection labels.

The English manuscript is intentionally not included yet. It will be added after translation and editorial review.

Run the lightweight integrity check from the repository root:

```bash
python scripts/smoke_test.py
```

It validates the result schema, checks that every recorded experiment has its analysis script, and reports the exact
number of bundled result files. It does not rerun GPU training.

## Repository layout

```text
experiments/cem/          CEM training, label controls, probes, and analyses
experiments/hicem/        HiCEM controls, configs, and analyses
experiments/dreamsim/     DreamSim label variants and evaluation scripts
experiments/goemotions/   GoEmotions/text controls and probe analyses
upstream/                 Minimal source snapshots of the three external codebases
results/                  Compact numerical outputs; no checkpoints or datasets
data/                     Data provenance and download instructions
environment/              Analysis and full-training dependency lists
scripts/                  Release and integrity checks
```

## Reproducing the reported analyses

### CEM on CUB-200-2011

The implementation is based on `mateoespinosa/cem` commit `26f027dd6f0c4231c68a3e3daa26b1ea9d35dbf8`. The relevant
entry points are:

* `experiments/cem/run_subsample_2022.py` — train the five CEM/Hybrid/control runs;
* `experiments/cem/make_placebo.py`, `make_placebo2.py`, `make_placebo3.py`, `make_placebo4.py` — construct label controls;
* `experiments/cem/probe_heldout.py` and `probe_variants.py` — held-out attribute probes;
* `experiments/cem/analyze_*.py` — aggregate accuracy, AP, and interval summaries.

The CUB images and Koh et al. `class_attr_data_10` files are required for a fresh run. They are not included here;
see `data/README.md`. Model checkpoints are also intentionally omitted. The checked-in result files allow a reviewer to
inspect the reported numbers without downloading the training artifacts.

### HiCEM / Concept Splitting

The source snapshot in `upstream/hicem/` comes from `OscarPi/cem-concept-discovery` commit
`4b665fa7a45803c76fb96506fcf44397283e668b`. The paper-specific wrappers are in `experiments/hicem/`:
`run_ctrl.py`, `hicem_checks3.py`, `null_match.py`, and the `cub_*.yaml` configurations. CUB split files and
Lightning checkpoints are not redistributed.

### DreamSim

The source snapshot in `upstream/dreamsim/` comes from `ssundaram21/dreamsim` commit
`db4d16c6948314e36e2e62f0bae2ee23bc974bf3`. The paper-specific scripts are `make_label_variants.py`, `eval_b.py`,
`summarize.py`, and the two job scripts. The NIGHTS, TID2013, and KADID-10k images and pretrained DINO/DreamSim
weights must be downloaded separately; see `data/README.md`.

### GoEmotions and text controls

`experiments/goemotions/` contains the complete set of small Python modules needed by the reported text analyses,
including the conditional probes, ridge probes, collapse checks, and spurious-control test. Raw text and embedding
caches are excluded; the public GoEmotions release is specified in `data/README.md`.

## Environment

For result inspection and CPU-only aggregation:

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r environment/requirements-analysis.txt
```

Full training additionally needs the packages listed in `environment/requirements-full.txt`, a CUDA-capable PyTorch
installation, and the upstream model-specific dependencies. Exact GPU drivers and wall-clock times are not treated as
part of the scientific result. The original execution host was a Linux server; no server path is required by the
scripts after data and model paths are configured.

## Scope and limitations

The numerical files are frozen snapshots of the runs used in the manuscript. The repository is a code-and-results
release; the English manuscript will be added in a later commit. Large datasets, model weights, checkpoints, and
training logs are intentionally excluded.

Third-party source snapshots retain their upstream license files where applicable. The license for the paper-specific
code and the final repository should be selected before the GitHub repository is made public; no license is silently
assigned here.
