# DeepLense — Physics-Informed Gravitational Lens Classification and Finding

This repository contains my ML4Sci DeepLense GSoC evaluation work across gravitational-lens classification, observational lens finding, and physics-guided modeling.

## Project focus

I worked on two DeepLense directions:

- **DEEPLENSE6 — Gravitational Lens Finding**
- **DEEPLENSE5 — Physics-Guided Machine Learning on Real Lensing Images**

The goal was to build strong PyTorch baselines, add physics-guided structure, evaluate robustness, and release reusable model artifacts.

## What I implemented

| Track | Task | What I built |
|---|---|---|
| Common Task I | Multi-class lens classification | DenseNet-based classifier for `no`, `sphere`, and `vort` classes |
| Specific Test V | Lens finding and data pipelines | Binary lens finder for 3-band observational cutouts |
| Specific Test VII | Physics-guided ML | Physics-informed / equivariant SSL experiments for lens classification |

## Results

| Component | Metric | Result |
|---|---:|---:|
| Physics-informed DenseNet | ROC-AUC | **0.9976** |
| Specific Test V lens finder | ROC-AUC | **0.9906** |
| Equivariant / physics SSL model | ROC-AUC | **0.9949** |

## Public artifacts

| Artifact | Link |
|---|---|
| Common Task / Specific Test VII weights | https://huggingface.co/datasets/Arushhh/physics_informed_densenet_common_task/tree/main |
| Specific Test V weights | https://huggingface.co/datasets/Arushhh/deeplense-test5-densepolar-artifacts/tree/main |
| Specific Test VII equivariant SSL weights | https://huggingface.co/datasets/Arushhh/eqv-phys-ssl-acpt-artificats/tree/main |

## Dataset setup

### Common Task I / Specific Test VII

The common-task dataset contains normalized strong-lensing images with three classes:

- `no`
- `sphere`
- `vort`

This setup was used for the DenseNet baseline and physics-guided extensions.

### Specific Test V

Specific Test V uses observational 3-band cutouts with shape `(3, 64, 64)`:

- `train_lenses`
- `train_nonlenses`
- `test_lenses`
- `test_nonlenses`

This task is closer to practical lens finding because it involves severe class imbalance and low false-positive requirements.

## Implementation details

- Framework: **PyTorch**
- Training environment: **Kaggle notebooks**
- GPU: **Tesla P100 16 GB**
- Main model families:
  - DenseNet-style CNNs
  - physics-informed variants
  - equivariant / self-supervised extensions
- Evaluation:
  - ROC-AUC
  - class-level validation
  - ablation comparisons
  - saved checkpoint artifacts

## Why this is not just a notebook dump

This repository is organized around reproducible research artifacts:

- task-specific notebooks
- model checkpoints
- result tables
- ablation writeups
- supporting figures
- public HuggingFace artifact release

## Reproducibility

To reproduce the experiments:

1. Open the corresponding task notebook.
2. Attach the required DeepLense dataset.
3. Run preprocessing cells.
4. Train the listed architecture.
5. Evaluate ROC-AUC.
6. Compare against the reported checkpoint.
7. Optionally load the public HuggingFace weights.

## Suggested repository structure

```text
Deeplense_GSoC/
├── common_task/
│   ├── notebooks/
│   ├── results/
│   └── figures/
├── specific_test_v/
│   ├── notebooks/
│   ├── results/
│   └── weights/
├── specific_test_vii/
│   ├── notebooks/
│   ├── results/
│   └── weights/
└── README.md
```

If the repo is not currently organized like this, use this structure as the cleanup target.

## Next improvements

- Add a one-command inference script for each released checkpoint.
- Add plots for ROC curves and confusion matrices.
- Add a `results/` folder with saved CSVs.
- Add a short paper-style method summary.
- Add links to exact Kaggle notebooks if they are public.

## Author

Arush Sharma
