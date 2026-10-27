# DEnet — modular refactor

This repository reorganizes the original [DEnet](https://github.com/KaitongZeng-web/DEnet) / Knowledge-Augmented GAT project into a conventional Python layout. The original repository has not been modified. The existing model and QM9 HOMO/LUMO task are retained; the primary change is organization and import-path consistency.

## Project structure

```text
DEnet-refactored/
├── knowledge_aug_gat/       # Reusable model/data package
│   ├── models/              # graph/image encoders, fusion, prediction head
│   ├── utils/               # QM9 dataset and SMILES processing
│   ├── constants.py
│   └── losses.py
├── engine/train.py          # training loop and smoke test
├── evaluation/test.py       # held-out test evaluation
├── inference/predict.py     # single-SMILES inference
├── tests/                   # lightweight structure/syntax checks
├── configs/default.yaml     # example configuration reference
├── scripts/                 # analysis scripts
├── legacy/                  # original single-file implementation
├── main.py                  # command-line entry point
└── requirements.txt
```

## Setup

Use a Python environment compatible with your PyTorch and PyTorch Geometric installation. PyTorch Geometric and RDKit installation can vary by operating system and CUDA version, so install the matching PyTorch build first if needed.

```bash
python -m pip install -r requirements.txt
```

## Commands

Run commands from the repository root:

```bash
# Quick forward/backward smoke test
python main.py --smoke-test

# Train (QM9 may be downloaded/processed on first run)
python main.py --train --epochs 100 --batch-size 32

# Evaluate a saved checkpoint
python main.py --test --checkpoint ./checkpoints/best.pt

# Predict HOMO/LUMO for a SMILES string
python main.py --predict --checkpoint ./checkpoints/best.pt --smiles CCO

# Run lightweight structure/syntax checks
python -m unittest discover -s tests -v
```

Data defaults to `./data/QM9`. Checkpoints are written to `./checkpoints/best.pt`. Large data and model checkpoints are intentionally excluded from Git.

## Important notes

- The current training code uses a simple fixed split: first 10,000 samples for training, next 1,000 for validation, next 1,000 for testing. This mirrors the original script and is not a randomized or benchmark-standard split.
- The SMILES inference code reconstructs input features. Confirm feature dimensions and checkpoint compatibility for your exact training setup before relying on predictions.
- This refactor has not been run through full model training or numerical validation in this environment. The smoke test requires all dependencies and QM9 data to be available.
- The yield-correlation plot is preserved as `scripts/plot_LUMO_Yield_Correlation.py`; it additionally requires pandas, SciPy and matplotlib.
