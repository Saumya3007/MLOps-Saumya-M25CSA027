# Assignment 4 - Neural Machine Translation with Transformer & Optuna

**Course:** Deep Learning Operations (DLOps)
**Author:** M25CSA027

---

## Overview
This assignment implements an **English-to-Hindi Neural Machine Translation (NMT)** system
using a Transformer (encoder-decoder) architecture trained from scratch. Hyperparameter
optimization is performed using **Optuna** with the TPE sampler and ASHA pruner.

---

## Dataset
- **English-Hindi parallel corpus** (~13,186 sentence pairs)
- Source: Kaggle Dataset (`saumyapanchole/english-hindi`)
- Vocabulary built with frequency threshold = 2
  - English vocab size: **4,117**
  - Hindi vocab size: **4,044**

---

## Model Architecture
| Component        | Details                          |
|------------------|----------------------------------|
| Model Type       | Transformer (Encoder-Decoder)    |
| d_model          | 512 (8 heads × 64)               |
| Feed-Forward Dim | 2048                             |
| Encoder Layers   | 4                                |
| Decoder Layers   | 3                                |
| Dropout          | 0.051                            |

---

## Hyperparameter Tuning (Optuna)
- **Sampler:** Tree-structured Parzen Estimator (TPE)
- **Pruner:** Successive Halving (ASHA)
- **Trials:** 15 (7 completed, 8 pruned)
- **Tuning Subset:** 5,000 samples, 20 epochs per trial
- **Best Trial:** Trial 14 — Val Loss: `1.6802`

### Best Hyperparameters
| Parameter      | Value               |
|----------------|---------------------|
| Learning Rate  | 2.015 × 10⁻⁴        |
| Batch Size     | 128                 |
| d_model        | 512                 |
| d_ff           | 2048                |
| Encoder Layers | 4                   |
| Dropout        | 0.051               |

---

## Results
| Model    | Epochs | Train Loss | BLEU Score |
|----------|--------|------------|------------|
| Baseline | 100    | 0.1484     | 0.5123     |
| Tuned    | 20     | 0.3500     | **0.5260** |

The tuned model achieves **higher BLEU** in **5× fewer epochs**.


---

## Requirements
torch, optuna, ray[tune], nltk, pandas, matplotlib, seaborn, scikit-learn,
huggingface_hub, accelerate, tqdm

text

Install via:
```bash
pip install torch optuna ray[tune] nltk pandas matplotlib seaborn scikit-learn huggingface_hub accelerate tqdm
