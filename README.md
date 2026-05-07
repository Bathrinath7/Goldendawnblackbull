# Chihuahua vs Muffin – Data-Centric AI with 3LC

## Overview

This repository contains our solution for the **Chihuahua vs Muffin Data-Centric AI Hackathon** using the **3LC platform** and a fixed **ResNet-18** architecture.

The objective of the competition was to improve classification accuracy **by improving the dataset**, not the model architecture.

We used:

- **3LC Dashboard**
- **Embeddings visualization**
- **Active learning**
- **Sample weighting**
- **Iterative labeling**
- **Data-centric AI workflows**

---

# Problem Statement

Build a binary image classifier:

| Label | Class |
|---|---|
| 0 | Chihuahua |
| 1 | Muffin |

Constraints:

- ResNet-18 only
- No pretrained weights
- No external datasets
- Data-centric improvements only

---

# Dataset

## Training Data

| Type | Count |
|---|---|
| Initially labeled | 100 |
| Unlabeled | 3,579 |

## Validation Data

| Type | Count |
|---|---|
| Validation images | 1,000 |

## Test Data

| Type | Count |
|---|---|
| Hidden test images | 1,184 |

---

# Data-Centric Workflow

Our workflow followed an iterative loop:

1. Train initial model on 100 labeled images
2. Generate embeddings and predictions using 3LC
3. Analyze clusters in 3D embedding space
4. Label high-confidence unlabeled samples
5. Enable samples using sample weights
6. Retrain model
7. Repeat until validation accuracy stabilized
8. Generate Kaggle submission

---

# Project Structure

```bash
.
├── data/
│   ├── train/
│   ├── val/
│   └── test/
│
├── register_tables.py
├── train.py
├── predict.py
├── config.yaml
├── submission.csv
├── best_model.pth
│
├── screenshots/
│   ├── embeddings.png
│   ├── dashboard_accuracy.png
│   └── clusters.png
│
├── writeup/
│   └── report.pdf
│
└── README.md
