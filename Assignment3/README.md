# Assignment 3 — End-to-End HuggingFace Model Training & Docker Deployment

Fine-tuning **DistilBERT** (`distilbert-base-uncased`) on the **Goodreads 8-Genre Book-Review** dataset for multi-class text classification, containerising the workflow using Docker, and publishing artifacts to HuggingFace Hub.

> **Colab run date:** 2026-02-20 · **Final Training Loss:** 2.0820 · **GPU:** Tesla T4

---

## 🔗 Quick Links

| Resource | Link |
|---|---|
| 🤗 HuggingFace Model | [Saumya3007/distilbert-goodreads-genre](https://huggingface.co/Saumya3007/distilbert-goodreads-genre) |
| 📊 WandB Report | [pancholisaumya-iit — hf-docker-assignment](https://api.wandb.ai/links/pancholisaumya-iit/z7mxf7zr) |

---

## 📁 Project Structure

```
assignment3/
├── Assignment3_Colab.ipynb     ← Phase 1: Run in Google Colab (Training + Upload)
├── src/
│   ├── __init__.py
│   ├── utils.py                ← Shared constants, GPU/CPU device detection
│   ├── data.py                 ← Dataset loading from HF Hub + tokenisation
│   ├── train.py                ← Fine-tuning with Trainer API + HF Hub upload
│   └── eval.py                 ← Evaluation (local + HF Hub) + visualisations
├── Dockerfile                  ← Dev image  : training + evaluation
├── Dockerfile.prod             ← Prod image : evaluation only, pulls model from HF Hub
├── requirements.txt            ← Full training + eval dependencies
├── requirements_prod.txt       ← Inference-only dependencies (CPU torch)
└── README.md
```

---

## 🐳 Docker — Build & Run

### 🔧 Development Image — Training + Evaluation

The dev image contains the full training stack with GPU support and all dev tools.

**Build:**
```bash
docker build -t assignment3-dev -f Dockerfile .
```

**Run training:**
```bash
docker run --rm   -e HF_TOKEN=<your_hf_write_token>   -e WANDB_API_KEY=<your_wandb_key>   -v $(pwd)/results:/app/results   assignment3-dev python -m src.train
```

**Run evaluation (after training):**
```bash
docker run --rm   -e HF_TOKEN=<your_hf_write_token>   -e WANDB_API_KEY=<your_wandb_key>   -e HF_REPO=Saumya3007/distilbert-goodreads-genre   -v $(pwd)/results:/app/results   assignment3-dev python -m src.eval
```

**Interactive shell inside container:**
```bash
docker run --rm -it assignment3-dev bash
```

---

### 🚀 Production Image — Evaluation Only

The production image uses a **multi-stage build**, runs as a **non-root user (`appuser`)**, contains **no training code**, uses **CPU-only PyTorch**, and pulls the model **live from HuggingFace Hub** on every container start.

**Build:**
```bash
docker build -t assignment3-prod -f Dockerfile.prod .
```

**Run (evaluation starts automatically on container boot):**
```bash
docker run --rm   -e HF_TOKEN=<your_hf_write_token>   -e HF_REPO=Saumya3007/distilbert-goodreads-genre   -v $(pwd)/results:/home/appuser/app/results   assignment3-prod
```

> `WANDB_MODE` is pre-set to `offline` in the prod image — no live WandB connection needed.

**Override `HF_REPO` without rebuilding:**
```bash
docker run --rm   -e HF_TOKEN=<your_hf_write_token>   -e HF_REPO=some_other_user/some-other-model   -v $(pwd)/results:/home/appuser/app/results   assignment3-prod
```

---


## 📊 Model Details

### Why DistilBERT?

| Property | Value |
|---|---|
| Model | `distilbert-base-uncased` |
| Total Parameters | 66,959,624 |
| Size vs BERT | 40% fewer parameters |
| Speed vs BERT | ~60% faster inference |
| GLUE performance | ~97% of BERT-base |
| Max sequence length | 256 tokens |
| Task | 8-class text classification |

DistilBERT was chosen because it fits within Colab T4 GPU memory (16 GB) at batch size 16, trains significantly faster than full BERT, and achieves strong classification performance. It is fully supported by the HuggingFace Trainer API with `id2label` / `label2id` mapping.

### Dataset

| Property | Value |
|---|---|
| Name | `adamnik/goodreads-genre-classification` |
| Task | 8-genre book review classification |
| Train samples | 144 |
| Validation samples | 16 |
| Test samples | 40 |
| Classes | `children`, `comics_graphic`, `fantasy_paranormal`, `history_biography`, `mystery_thriller_crime`, `poetry`, `romance`, `young_adult` |

### Training Configuration

| Hyperparameter | Value |
|---|---|
| Epochs | 3 |
| Batch size | 16 |
| Learning rate | 2e-5 |
| Warmup steps | 200 |
| Weight decay | 0.01 |
| Optimizer | AdamW |
| Eval strategy | Every 100 steps |
| Save strategy | Every 100 steps |
| Best model metric | F1 (weighted) |
| Early stopping patience | 3 |
| FP16 | ✅ Enabled (GPU) |
| Seed | 42 |

---

## 📈 Results

### Training Loss Curve

| Step | Training Loss | Validation Loss |
|---|---|---|
| 100 | — | — |
| Final | **2.0820** | 2.0826 |

> Full step-by-step loss curve available on WandB → [run jtgvl5ef](https://api.wandb.ai/links/pancholisaumya-iit/z7mxf7zr)

### Evaluation Results

| Metric | Local Model | HuggingFace Model | Δ |
|---|---|---|---|
| **eval_loss** | 2.0826 | 2.0826 | 0.0000 |
| **eval_accuracy** | 0.1250 | 0.1250 | 0.0000 |
| **eval_f1 (weighted)** | 0.0284 | 0.0284 | 0.0000 |
| eval_runtime (s) | 0.1385 | 0.5731 | — |
| eval_samples/sec | 288.864 | 69.794 | — |

> Identical scores between local and HuggingFace Hub model confirm successful serialisation and upload. The low accuracy (0.125) and F1 (0.028) reflect the very small dataset size (144 train samples across 8 classes ≈ 18 samples/class).

### Generated Visualisations

All plots saved to `results/` and auto-logged to WandB:

| File | Description |
|---|---|
| `results/dataset_eda.png` | Genre distribution bar chart + review length histogram |
| `results/training_history.png` | 2×2 grid: train loss, val loss, val accuracy, val F1 over steps |
| `results/cm_local.png` | Confusion matrix — local model |
| `results/cm_hf.png` | Confusion matrix — HuggingFace Hub model |
| `results/f1_local.png` | Per-class F1 bar chart — local model |
| `results/f1_hf.png` | Per-class F1 bar chart — HuggingFace Hub model |
| `results/comparison.png` | Local vs HuggingFace metric comparison bar chart |
| `results/local_metrics.json` | Local model eval metrics (JSON) |
| `results/hf_metrics.json` | HuggingFace model eval metrics (JSON) |

---

## 📝 Short Report

### Model Selection
DistilBERT (`distilbert-base-uncased`) was selected over full BERT for its 40% smaller parameter count (66.9M vs 110M) and 60% faster inference while retaining ~97% of BERT's GLUE benchmark performance. This made it ideal for Colab's T4 GPU within time and memory constraints for a multi-class classification task.

### Training Summary
The model was fine-tuned for 3 epochs on the Goodreads 8-genre book review dataset using the HuggingFace Trainer API with AdamW optimiser, early stopping (patience=3), FP16 mixed precision, and evaluation every 100 steps. The final training loss was **2.0820**. Training ran on a Tesla T4 GPU on Google Colab.

### Evaluation Comparison
Both the locally saved model and the model reloaded from HuggingFace Hub (`Saumya3007/distilbert-goodreads-genre`) produced **identical scores** (accuracy=0.1250, F1=0.0284), confirming successful serialisation and upload. The low metrics are expected given the small dataset — only ~18 training samples per class with 8 genres is insufficient for meaningful convergence.

### Challenges
- Very small dataset (144 train samples / 8 classes) leading to low accuracy — real improvement would require the full dataset without `SAMPLE_SIZE` cap
- `evaluation_strategy` → `eval_strategy` deprecation fix needed in newer `transformers` versions (applied in notebook)
- `COPY __init__.py ./` Docker build error — resolved by removing the line (only `src/__init__.py` is needed, already included via `COPY src/ ./src/`)
- Production Docker multi-stage build required careful separation of builder and runtime layers to keep the final image lean
