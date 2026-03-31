# GSoC 2026 DeepLense Evaluation Tasks

This repository contains my evaluation-task work for the following ML4SCI GSoC 2026 DeepLense project directions:

- **DEEPLENSE6 — Gravitational Lens Finding**  
  https://ml4sci.org/gsoc/2026/proposal_DEEPLENSE6.html
- **DEEPLENSE5 — Physics Guided Machine Learning on Real Lensing Images**  
  https://ml4sci.org/gsoc/2026/proposal_DEEPLENSE5.html

## What is included

For **DEEPLENSE6**, I completed:
- **Common Task I** — Multi-Class Classification
- **Specific Test V** — Lens Finding & Data Pipelines

For **DEEPLENSE5**, I completed:
- **Common Task I** — Multi-Class Classification
- **Specific Test VII** — Physics-Guided ML

## Implementation setup

All tasks were implemented in **PyTorch** using **Kaggle notebooks** on a **Tesla P100 16 GB GPU**.

I used Kaggle because it gave me:
- a simple notebook-based workflow for rapid experimentation,
- a reproducible GPU environment,
- and an easy way to train, evaluate, save checkpoints, and organize outputs for multiple ablations.

## Dataset setup

### Common Task I / Specific Test VII
These tasks use the same normalized strong-lensing image dataset with three classes:
- `no`
- `sphere`
- `vort`

My work here focuses on:
- a strong common-task benchmark,
- and more physics-guided / PINN-style extensions for Specific Test VII.

### Specific Test V
This task uses observational **3-band cutouts** with shape `(3, 64, 64)` for binary lens finding:
- `train_lenses`
- `train_nonlenses`
- `test_lenses`
- `test_nonlenses`

My work here focuses on:
- a strong benchmark lens finder,
- candidate ranking under severe class imbalance,
- and research directions for reducing false positives.

## Why I worked on these tasks

I chose these tasks because together they cover two parts of the DeepLense problem that I find especially interesting:

- **strong classification baselines and physics-guided modeling**
- **practical lens finding on observational data with low false-positive requirements**

The common task gave me a clean starting point for benchmarking architectures, while Specific Test V and Specific Test VII let me explore two directions I care about more deeply:
- lens finding in realistic observational settings,
- and physics-guided learning through more structured model design.

## Repository note

This repository is organized as a collection of task-specific notebooks, ablations, write-ups, and supporting figures.

## Model weights

- **Common Task / Specific Test VII weights:** [Add link here]
- **Specific Test V weights:** [Add link here]
- - **Specific Test VII weights:** [Add link here]

## Author

**Arush Sharma**
