# 🧠 Multitask Spatiotemporal GNN for Interpretable Prediction of PACC and Conversion Time in Preclinical Alzheimer
## 📌 Overview

MAGNET-AD is a novel multitask spatiotemporal graph neural network (STGNN) designed to predict both the Preclinical Alzheimer's Cognitive Composite (PACC) score and time to AD conversion. It achieves state-of-the-art performance by integrating multimodal data and capturing the complex interplay of biological, structural, and temporal factors in preclinical Alzheimer's Disease.

This repository contains the official inference code for the MAGNET-AD (Multitask Spatiotemporal GNN for Interpretable Prediction of PACC and Conversion Time in Preclinical Alzheimer) framework.


## Architecture

![Model Architecture](Figures/MAGNET_AD_Arch.png)

The framework consists of four key components:

1. 🧩  **Hybrid Data Fusion** : Integrates dynamic neuroimaging patterns with time-invariant genetic markers through weighted edges
2. ⏱️  **Dual Attention Mechanisms** : Employs spatial attention for relationships between brain structures and genetic factors, and temporal attention for structural changes across visits
3. 📊  **Multi-Task Learning** : Simultaneously predicts PACC scores and AD conversion time through specialized prediction heads
4. 📈  **Temporal Importance Weighting** : Adaptively learns critical time points in disease progression using an innovative loss function

