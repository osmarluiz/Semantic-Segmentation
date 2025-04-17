# Semantic-Segmentation: Sparse Annotation with Iterative Active Learning

![License](https://img.shields.io/badge/license-MIT-green)

This repository implements the framework proposed in the paper:

**"Do We Need Dense Labels? A Framework for Sparse Annotation and Iterative Active Learning"**  
Osmar Luiz Ferreira de Carvalho, Osmar Abílio de Carvalho Júnior, Anesmar Olino de Albuquerque, Daniel Guerreiro e Silva  
*Submitted to IEEE Transactions on Geoscience and Remote sensing

## 🚀 Overview

This framework enables training, collecting new points, and retraining in a loop to achieve high performance on semantic segmentation with few annotations.
- Sparse point-level annotations
- Iterative active learning
- A novel Dynamic Weighted Confidence Dice Loss (a loss designed to assign larger weights to misclassified pixels and lower weights to easy pixels)
- An intuitive Jupyter-based interactive UI

## 🔁 Methodology

We propose a complete human-in-the-loop pipeline with training, annotation, and retraining cycles. The users can select points at any position but we experimented by selecting error-prone regions and updates the model.

**Workflow overview (Fig. 3):**

![Methodological Flowchart](figs/fig3.png) <!-- Replace with actual path in repo -->

---

## 🖥️ Software Interface

The system provides an intuitive UI to support annotation, visualization, and interaction.

**Software UI (Fig. 4):**

![Software Interface](figs/fig4.png) <!-- Replace with actual path in repo -->

Features include:
- Click-based sparse annotation
- Real-time overlay of predictions
- Class toggles, point editing, and cursor data
- Iteration tracking and metric logging
- Full support for binary and multiclass segmentation

## 📊 Results

The framework achieves up to **96.9%** of the dense annotation performance using less than **0.02%** labeled pixels, across tasks like:
- Car, road, building, and permeable surface segmentation
- Binary and multiclass setups

---

## 📁 Dataset

Experiments are based on a high-resolution aerial dataset from Brasília (256×256 RGB patches). The dataset includes:
- 1,000 training patches
- 250 validation patches
- 250 test patches

To have access to the dataset, please contact osmarcarvalho@ieee.org
