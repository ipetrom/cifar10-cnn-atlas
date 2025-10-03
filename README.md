_the project was developed in December 2024_


# CIFAR-10 CNN (Atlas) — Notebook Overview

This repository contains a Jupyter Notebook that implements a convolutional neural network (CNN) trained on the CIFAR-10 dataset. The notebook demonstrates a full workflow from data loading and preprocessing, through model definition and training, to evaluation and visualization.
---

## Project summary

- Dataset: CIFAR-10 (60,000 color images, 32×32, 10 classes)
- Task: Image classification (10 classes)
- Model: Custom CNN (`Net2`) implemented in PyTorch
- Key features:
  - Computes dataset mean and standard deviation for normalization
  - Data augmentation (random horizontal flips and rotations)
  - Train / validation / test split
  - Training loop with early stopping and best-weight checkpointing
  - Evaluation and per-class accuracy reporting
  - Visualizations for samples and training progress
  - Model saving (PyTorch)

## Model architecture (high level)

- Conv blocks:
  - 3 → 32 → BatchNorm → ReLU
  - 32 → 64 → BatchNorm → ReLU → MaxPool
  - 64 → 128 → BatchNorm → ReLU
  - 128 → 128 → BatchNorm → ReLU → MaxPool (outputs 128×8×8)
  - 128 → 256 → BatchNorm → ReLU
  - 256 → 256 → BatchNorm → ReLU → MaxPool (outputs 256×4×4)
- Classifier:
  - Flatten (256×4×4 → 4096)
  - Linear 4096 → 512 → ReLU → Dropout
  - Linear 512 → 256 → ReLU
  - Linear 256 → 10 (logits)

Total parameters (from a sample run): ~3.36M

---

## Training details

- Loss: CrossEntropyLoss
- Optimizer: Adam (learning rate = 1e-3)
- Epochs: up to 100 (early stopping used)
- Early stopping: patience = 10 (best validation loss tracked)
- Batch size: 32 (train / val / test)
- Device: GPU if available (CUDA), otherwise CPU
- Example training outcomes in the notebook run:
  - Validation accuracy reached > 98% (peak in the run ~99.05% reported)
  - Training stopped early around epoch 92

Note: results depend on random seed, augmentation, exact splitting and environment (GPU/CPU).

---

## Data preprocessing & augmentation

- Mean/std computed over the training set (example values printed in the notebook):
  - mean ≈ [0.4914, 0.4822, 0.4465]
  - std  ≈ [0.2023, 0.1994, 0.2010]
- Training transforms:
  - RandomHorizontalFlip
  - RandomRotation(20)
  - ToTensor()
  - Normalize(mean, std)
- Validation/test transforms:
  - ToTensor()
  - Normalize(mean, std)

The notebook also demonstrates visualizations of raw and transformed image batches using torchvision utilities and matplotlib.

---

## Libraries / technologies used

- Python
- PyTorch (torch, torchvision)
- NumPy
- Matplotlib, Seaborn (visualizations)
- pandas
- scikit-learn (metrics)
- mlxtend (confusion matrix plotting)
- torchinfo (model summary)
- Jupyter Notebook / Kaggle environment 

---