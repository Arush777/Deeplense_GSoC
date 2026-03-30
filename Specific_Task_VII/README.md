# DeepLense Specific Test VII — Physics-Guided ML

PyTorch notebook series for **DeepLense GSoC Specific Test VII: Physics-Guided ML**.

This repository studies a staged family of **physics-informed neural networks (PINNs)** for classifying strong gravitational lensing images into three classes:

- `no`
- `sphere`
- `vort`

The dataset is the same three-class lensing dataset used in the Common Task.  
The goal here is not only to classify images well, but to make the architecture reason through quantities tied to the **gravitational lensing equation** instead of treating the task as ordinary texture classification.

---

## Task Summary

Specific Test VII asks for a model that uses the **gravitational lensing equation** inside the architecture and improves the scientific quality of the classifier through physics-guided reasoning.

In this repository, I approached the task as a **research progression** rather than a single-model leaderboard submission:

1. establish strong dense physics-guided baselines,
2. move toward more explicit field-based PINN designs,
3. test a compact equivariant supervised architecture,
4. then extend that family into a physics-guided self-supervised representation learning setup.

Because of that, this repository contains both:
- **strong benchmark-style notebook baselines**, and
- **lighter, more interpretable, more research-aligned equivariant PINN tiers**.

---

## Main Framing of the Repository

This repository is best understood in three layers:

### 1. Strong dense baselines
These are the strongest raw-performing Task VII notebook runs and serve as reference points for the whole study.

### 2. Main supervised physics-guided architecture
This is the **Tier 2 EQV-AC-PT-PINN** model, which I treat as the main architecture to center the proposal around.

### 3. Future-facing research extension
This is the **Tier 3 EQV-PHYS-SSL-ACPT** notebook, which extends the same architecture family into **physics-guided self-supervised pretraining** and supervised finetuning.

---

## Notebook Progression

| Notebook | Role in repository | Split handling | Accuracy | Macro AUC |
|---|---|---|---:|---:|
| `task-vii-dataset_best_auc.ipynb` | Strong DenseNet residual-PINN benchmark | rebuilt fresh 90:10 from full dataset | 0.9701 | 0.9969 |
| `task-vii_second_best_auc.ipynb` | Strong DenseNet PINN benchmark | rebuilt fresh 90:10 from full dataset | 0.9715 | 0.9966 |
| `ac-pt-pinn.ipynb` | ConvNeXt + attention-consistency physics PINN ablation | provided `train/` and `val/` split | 0.9688 | 0.9959 |
| `eqv-lenspinn-abilation_I.ipynb` | Tier 1 equivariant feasibility baseline | provided `train/` and `val/` split | 0.9448 | 0.9917 |
| `eqv-ac-pt-pinn-abilation_II.ipynb` | **Tier 2 main supervised contribution** | provided `train/` and `val/` split | 0.9557 | 0.9931 |
| `eqv-phys-ssl-acpt.ipynb` | Tier 3 physics-guided SSL extension | provided `train/` and `val/` split | 0.9603 | 0.9949 |

### Important note on comparison
The notebooks do **not** all use exactly the same split protocol:
- the two DenseNet-style Task VII benchmark notebooks rebuild a fresh **90:10** split from the full dataset,
- while the later equivariant tier notebooks mostly use the provided `train/` and `val/` folders directly.

So the progression should be read as a **careful notebook audit and staged architecture study**, not as a perfectly apples-to-apples leaderboard table.

---

## Repository Structure

```text
.
├── README.md
├── assets/
│   ├── task7-tier2-eqv-ac-pt-pinn.png
│   ├── task7-residual-descriptors.png
│   └── task7-tier3-eqv-phys-ssl-acpt.png
├── task-vii-dataset_best_auc.ipynb
├── task-vii_second_best_auc.ipynb
├── ac-pt-pinn.ipynb
├── eqv-lenspinn-abilation_I.ipynb
├── eqv-ac-pt-pinn-abilation_II.ipynb
├── eqv-phys-ssl-acpt.ipynb
└── requirements.txt
