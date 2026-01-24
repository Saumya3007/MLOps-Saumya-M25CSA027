<div align="center">

# 🧠 Deep Learning Image Classification
### ResNet vs SVM on MNIST & FashionMNIST

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Report](https://img.shields.io/badge/Report-PDF-orange.svg)](assignment1/report.pdf)

**[📄 Full Report](https://github.com/Saumya3007/MLOps-Saumya-M25CSA027/blob/main/Assignment1/M25CSA027_Saumya_Pancholi_ass1.pdf)** | **[💾 Trained Models](https://drive.google.com/drive/folders/1-dOCRC0NZbTkfPeybTe9rsJcY7rmiUC7?usp=sharing)** 

</div>

---

## 📋 Table of Contents
- [Overview](#-overview)
- [Q1(a): ResNet Experiments](#qa-resnet-deep-learning-experiments)
- [Q1(b): SVM Experiments](#qb-svm-classification)
- [Q2: CPU vs GPU Analysis](#q2-cpu-vs-gpu-performance)
- [Results Summary](#-results-summary)
- [Setup & Usage](#-setup--usage)

---

## 🎯 Overview

<p align="justify">
<sub>This project presents a comprehensive experimental analysis comparing deep learning (ResNet-18/50) and traditional machine learning (SVM) approaches for image classification on MNIST and FashionMNIST datasets. We evaluate multiple hyperparameter configurations, analyze CPU vs GPU performance, and provide detailed insights into model selection for production deployment.</sub>
</p>

### 🗂️ Datasets
| Dataset | Images | Size | Classes | Split Ratio |
|---------|--------|------|---------|-------------|
| **MNIST** | 60,000 | 28×28 | 10 (Digits) | 70/10/20 |
| **FashionMNIST** | 60,000 | 28×28 | 10 (Clothing) | 70/10/20 |

<sub>*Split: Train/Validation/Test*</sub>

---

## Q1(a): ResNet Deep Learning Experiments

<p align="justify">
<sub>Trained ResNet-18 and ResNet-50 architectures with systematic hyperparameter exploration including batch sizes (16, 32), optimizers (SGD, Adam), learning rates (0.001, 0.0001), epochs (2, 3), and pin_memory configurations. All experiments utilized mixed precision training (AMP) for efficiency.</sub>
</p>

### 🏆 Best Configurations

<table>
<thead>
<tr>
<th><sub>Dataset</sub></th>
<th><sub>Model</sub></th>
<th><sub>Batch</sub></th>
<th><sub>Optimizer</sub></th>
<th><sub>LR</sub></th>
<th><sub>Epochs</sub></th>
<th><sub>Test Acc (%)</sub></th>
</tr>
</thead>
<tbody>
<tr>
<td><sub><b>MNIST</b></sub></td>
<td><sub>ResNet-18</sub></td>
<td><sub>16</sub></td>
<td><sub>SGD</sub></td>
<td><sub>0.001</sub></td>
<td><sub>2</sub></td>
<td><sub><b>99.01</b></sub></td>
</tr>
<tr>
<td><sub><b>MNIST</b></sub></td>
<td><sub>ResNet-50</sub></td>
<td><sub>16</sub></td>
<td><sub>SGD</sub></td>
<td><sub>0.001</sub></td>
<td><sub>3</sub></td>
<td><sub>98.27</sub></td>
</tr>
<tr>
<td><sub><b>FashionMNIST</b></sub></td>
<td><sub>ResNet-18</sub></td>
<td><sub>32</sub></td>
<td><sub>Adam</sub></td>
<td><sub>0.001</sub></td>
<td><sub>3</sub></td>
<td><sub><b>89.24</b></sub></td>
</tr>
<tr>
<td><sub><b>FashionMNIST</b></sub></td>
<td><sub>ResNet-50</sub></td>
<td><sub>16</sub></td>
<td><sub>SGD</sub></td>
<td><sub>0.001</sub></td>
<td><sub>3</sub></td>
<td><sub>86.02</sub></td>
</tr>
</tbody>
</table>



---

## Q1(b): SVM Classification

<p align="justify">
<sub>Evaluated Support Vector Machines with RBF and Polynomial kernels across various hyperparameter configurations. Tested C values (0.1, 1.0, 10.0, 100.0), gamma settings ('scale', 0.001, 0.01), and polynomial degrees (2, 3). Features were flattened to 784 dimensions and normalized using StandardScaler.</sub>
</p>

### 🏆 Best Configurations

<table>
<thead>
<tr>
<th><sub>Dataset</sub></th>
<th><sub>Kernel</sub></th>
<th><sub>C</sub></th>
<th><sub>γ</sub></th>
<th><sub>Degree</sub></th>
<th><sub>Train Acc (%)</sub></th>
<th><sub>Test Acc (%)</sub></th>
<th><sub>Time (sec)</sub></th>
</tr>
</thead>
<tbody>
<tr>
<td><sub><b>MNIST</b></sub></td>
<td><sub>Polynomial</sub></td>
<td><sub>1.0</sub></td>
<td><sub>0.01</sub></td>
<td><sub>3</sub></td>
<td><sub>100.0</sub></td>
<td><sub><b>97.39</b></sub></td>
<td><sub>386.26</sub></td>
</tr>
<tr>
<td><sub><b>FashionMNIST</b></sub></td>
<td><sub>RBF</sub></td>
<td><sub>10.0</sub></td>
<td><sub>scale</sub></td>
<td><sub>-</sub></td>
<td><sub>98.4</sub></td>
<td><sub><b>89.27</b></sub></td>
<td><sub>486.14</sub></td>
</tr>
</tbody>
</table>


---

## Q2: CPU vs GPU Performance

<p align="justify">
<sub>Comprehensive comparison of training performance between CPU and GPU environments on FashionMNIST dataset. Evaluated both ResNet architectures with SGD and Adam optimizers, measuring training time, accuracy, and computational efficiency (FLOPs).</sub>
</p>

### ⚡ Speed Comparison

<table>
<thead>
<tr>
<th rowspan="2"><sub>Model</sub></th>
<th rowspan="2"><sub>Optimizer</sub></th>
<th colspan="2"><sub>Training Time (sec)</sub></th>
<th rowspan="2"><sub>Speedup</sub></th>
<th rowspan="2"><sub>Time Saved</sub></th>
</tr>
<tr>
<th><sub>CPU</sub></th>
<th><sub>GPU</sub></th>
</tr>
</thead>
<tbody>
<tr>
<td><sub>ResNet-18</sub></td>
<td><sub>SGD</sub></td>
<td><sub>794.53</sub></td>
<td><sub>445.55</sub></td>
<td><sub><b>1.78×</b></sub></td>
<td><sub>5.8 min</sub></td>
</tr>
<tr>
<td><sub>ResNet-18</sub></td>
<td><sub>Adam</sub></td>
<td><sub>853.10</sub></td>
<td><sub>322.71</sub></td>
<td><sub><b>2.64×</b></sub></td>
<td><sub>8.8 min</sub></td>
</tr>
<tr>
<td><sub>ResNet-50</sub></td>
<td><sub>SGD</sub></td>
<td><sub>1788.51</sub></td>
<td><sub>517.36</sub></td>
<td><sub><b>3.46×</b></sub></td>
<td><sub>21.2 min</sub></td>
</tr>
<tr>
<td><sub>ResNet-50</sub></td>
<td><sub>Adam</sub></td>
<td><sub>2070.10</sub></td>
<td><sub>539.41</sub></td>
<td><sub><b>3.84×</b></sub></td>
<td><sub>25.5 min</sub></td>
</tr>
</tbody>
</table>

|
---

## 📊 Results Summary

### 🏅 Cross-Method Comparison

<table>
<thead>
<tr>
<th><sub>Method</sub></th>
<th><sub>MNIST Acc (%)</sub></th>
<th><sub>FashionMNIST Acc (%)</sub></th>
<th><sub>Training Time</sub></th>
<th><sub>Best Use Case</sub></th>
</tr>
</thead>
<tbody>
<tr>
<td><sub><b>ResNet-18</b></sub></td>
<td><sub><b>99.01</b></sub></td>
<td><sub><b>89.24</b></sub></td>
<td><sub>0.7 min</sub></td>
<td><sub>Production deployment</sub></td>
</tr>
<tr>
<td><sub>ResNet-50</sub></td>
<td><sub>98.27</sub></td>
<td><sub>86.02</sub></td>
<td><sub>1.8 min</sub></td>
<td><sub>Complex patterns</sub></td>
</tr>
<tr>
<td><sub>SVM (Poly)</sub></td>
<td><sub>97.39</sub></td>
<td><sub>88.65</sub></td>
<td><sub>6.4 min</sub></td>
<td><sub>Interpretability</sub></td>
</tr>
<tr>
<td><sub>SVM (RBF)</sub></td>
<td><sub>96.90</sub></td>
<td><sub>89.27</sub></td>
<td><sub>8.1 min</sub></td>
<td><sub>Baseline comparison</sub></td>
</tr>
</tbody>
</table>

### 🔑 Key Insights

<sub>

- ✅ **ResNet-18 outperforms ResNet-50** on both datasets despite having 54% fewer parameters
- ⚡ **GPU acceleration essential**: 1.78-3.84× speedup with negligible accuracy difference
- 🎯 **SVM remains competitive**: Achieves 89.27% on FashionMNIST, matching ResNet-18
- 📈 **Optimizer matters**: SGD optimal for MNIST, Adam superior on FashionMNIST
- 🚀 **ResNet-18 is production-ready**: Best accuracy-speed trade-off for deployment

</sub>

---

## 🛠️ Setup & Usage

### Prerequisites

<sub>

```bash
Python 3.8+
PyTorch 2.0+
torchvision
scikit-learn
pandas
matplotlib
seaborn
