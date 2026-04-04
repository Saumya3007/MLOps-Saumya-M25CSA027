# Assignment 5 — ViT-S LoRA Fine-Tuning on CIFAR-100

> **HuggingFace Model:** https://huggingface.co/Saumya3007/vit-s-cifar100-lora-best
> 
> **WandB Report:** https://wandb.ai/pancholisaumya-iit/Q1_Assignment1_vit-cifar100-lora/reports/Q1_Wandb-Run-report--VmlldzoxNjQyMTUwMQ

---

## Setup

```bash
# 1. Clone and switch to branch
git clone https://github.com/<your-repo>.git
cd <your-repo>
git checkout "Assignment 5"

# 2. Fill credentials
cp .env.example .env
# Edit .env → set HF_USERNAME, HF_TOKEN, WANDB_API_KEY

# 3. Build & run Docker
docker build -t ass5 .
docker run --gpus all -v $(pwd):/app ass5
```

Or without Docker:
```bash
pip install -r requirements.txt
python main.py
```

---

## Run Training

```bash
# Full pipeline (baseline + LoRA grid + Optuna)
python main.py

# Skip grid search
python main.py --skip_grid

# Skip Optuna
python main.py --skip_optuna

# Custom epochs
python main.py --epochs 5
```

---

## Project Structure
assignment5/
├── main.py # Full pipeline entry point
├── src/
│ ├── config.py # All hyperparameters
│ ├── dataset.py # CIFAR-100 dataloaders
│ ├── model.py # ViT-S baseline + LoRA model builders
│ ├── trainer.py # Train/eval loops + WandB logging
│ ├── plots.py # Loss/acc curves, heatmap, classwise histogram
│ └── upload.py # HuggingFace Hub upload
├── weights/ # Saved model checkpoints
├── results/ # JSON history + test summary table
├── plots/ # All generated figures
├── Dockerfile
├── requirements.txt
└── .env.example

text

---

## Results

| Experiment | LoRA | Rank | Alpha | Dropout | Test Acc | Trainable Params |
|---|---|---|---|---|---|---|
| baseline_no_lora | without | -- | -- | -- | 81.76% | 76,900 |
| lora_r2_a2 | with | 2 | 2 | 0.1 | 90.12% | 75,364 |
| lora_r2_a4 | with | 2 | 4 | 0.1 | 90.16% | 75,364 |
| lora_r2_a8 | with | 2 | 8 | 0.1 | 90.00% | 75,364 |
| lora_r4_a2 | with | 4 | 2 | 0.1 | 90.21% | 112,228 |
| lora_r4_a4 | with | 4 | 4 | 0.1 | 90.41% | 112,228 |
| lora_r4_a8 | with | 4 | 8 | 0.1 | 90.45% | 112,228 |
| lora_r8_a2 | with | 8 | 2 | 0.1 | 90.20% | 185,956 |
| lora_r8_a4 | with | 8 | 4 | 0.1 | 90.60% | 185,956 |
| lora_r8_a8 | with | 8 | 8 | 0.1 | 90.37% | 185,956 |
| **optuna_best** | **with** | **16** | **16** | **0.0** | **90.69%** | **333,412** |

**Optuna Best Params:** rank=16, alpha=16, dropout=0.0, lr=0.001393

---