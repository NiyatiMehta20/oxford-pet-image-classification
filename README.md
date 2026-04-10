# Oxford-IIIT Pet Image Classification

> Multi-model image classification pipeline for the Oxford-IIIT Pet Dataset (37 breeds) using PyTorch — comparing a custom CNN, ResNet-18 transfer learning, data augmentation, and a hierarchical cat/dog classifier.


---

## Overview

This assignment benchmarks four increasingly sophisticated deep learning approaches for fine-grained pet breed classification across 37 classes (12 cat breeds, 25 dog breeds) using the Oxford-IIIT Pet Dataset (~7,300 images).

---

## Models Implemented

| Task | Model | Test Accuracy |
|------|-------|--------------|
| Task 2 | SimpleCNN (AlexNet-style, from scratch) | ~4.25% |
| Task 3 | ResNet-18 Transfer Learning | **86.37%** |
| Task 4 | ResNet-18 + Data Augmentation | 84.90% |
| Task 5 | Hierarchical ResNet-18 (binary → fine-grained) | 81.25% (fine-grained) / 98.20% (cat vs dog) |

---

## Project Structure
├── Niyati_25202110_assignment_1_pets.ipynb   # Main notebook (all tasks)
├── Niyati_25202110_assignment_1_pets.pdf     # Exported PDF report
└── README.md

---

## Tasks Breakdown

- **Task 1** — Dataset setup and sample image visualisation
- **Task 2** — Custom AlexNet-style CNN trained from scratch
- **Task 3** — Transfer learning with pre-trained ResNet-18 (ImageNet weights, frozen backbone)
- **Task 4** — Data augmentation (RandomResizedCrop, HorizontalFlip, ColorJitter) applied to Task 3
- **Task 5** — Hierarchical classifier: binary (cat/dog) head + fine-grained breed head on shared ResNet-18 backbone
- **Task 6** — Full test set evaluation across all models
- **Task 7** — Written reflection on model design choices and results

---

## Key Results

- Transfer learning dramatically outperformed the from-scratch CNN (86% vs 4%), confirming the value of ImageNet pretraining on small datasets
- Data augmentation improved generalisation but marginally underperformed the non-augmented transfer model on test data
- The hierarchical model achieved 98.20% binary accuracy (cat vs dog) and 81.25% fine-grained accuracy, showing error propagation between stages
- Best overall model: **ResNet-18 with Transfer Learning** (Task 3)

---

## Tech Stack

`Python` `PyTorch` `Torchvision` `ResNet-18` `NumPy` `Matplotlib` `Google Colab` `CUDA`

---

## Dataset

[Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/) — automatically downloaded via `torchvision.datasets.OxfordIIITPet`

- Train/Val: 3,680 images (80/20 split → 2,944 train / 736 val)
- Test: 3,669 images
- Classes: 37 breeds
