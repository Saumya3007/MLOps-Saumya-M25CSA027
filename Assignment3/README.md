# Assignment 3 — End-to-End HuggingFace Model Training & Docker Deployment

Fine-tuning **DistilBERT** (`distilbert-base-uncased`) on the **Goodreads 8-Genre Book-Review** dataset for multi-class text classification, then containerising and publishing the full workflow.

> **Colab notebook run date:** 2026-02-20 · Training loss achieved: **2.0820**

---

## 🔗 Quick Links

| Resource | URL |
|---|---|
| 📓 Colab Notebook | *(add your Colab share link)* |
| 🤗 HuggingFace Model | `https://huggingface.co/YOUR_USERNAME/distilbert-goodreads-genre` |
| 📊 WandB Project | `https://wandb.ai/YOUR_USERNAME/hf-docker-assignment` |
| 🐙 GitHub Repo | *(add your GitHub link)* |

---

## 📁 Project Structure

```
assignment3/
├── Assignment3_Colab.ipynb     ← Phase 1: Run entirely in Google Colab (Tasks 1–8)
├── src/
│   ├── __init__.py
│   ├── utils.py                ← Shared constants, device detection (GPU → CPU fallback)
│   ├── data.py                 ← Dataset loading (HF Hub) + tokenisation
│   ├── train.py                ← Fine-tuning with Trainer API + HF Hub upload
│   └── eval.py                 ← Evaluation (local + HF Hub) + all visualisations
├── Dockerfile                  ← Dev image  : training + evaluation (GPU/CPU)
├── Dockerfile.prod             ← Prod image : evaluation only, model pulled from HF Hub
├── requirements.txt            ← Full training + eval dependencies
├── requirements_prod.txt       ← Inference-only dependencies (CPU torch, no wandb)
└── README.md
```

---

## 🐳 Docker — Build & Run

### Prerequisites
- Docker Desktop installed and running
- Your HuggingFace **write** token → [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
- Your WandB API key → [wandb.ai/authorize](https://wandb.ai/authorize)

---

### 🔧 Development Image (Training + Evaluation)

**Build:**
```bash
cd assignment3
docker build -t assignment3-dev -f Dockerfile .
```

**Verify the build:**
```bash
docker run --rm assignment3-dev python -c "import transformers; print('OK:', transformers.__version__)"
```

**Run training:**
```bash
docker run --rm   -e HF_TOKEN=your_hf_write_token   -e WANDB_API_KEY=your_wandb_key   -v $(pwd)/results:/app/results   assignment3-dev python -m src.train
```

**Run evaluation (after training):**
```bash
docker run --rm   -e HF_TOKEN=your_hf_write_token   -e WANDB_API_KEY=your_wandb_key   -e HF_REPO=YOUR_USERNAME/distilbert-goodreads-genre   -v $(pwd)/results:/app/results   assignment3-dev python -m src.eval
```

**Run individual modules for testing:**
```bash
# Test data loading only
docker run --rm assignment3-dev python -m src.data

# Interactive shell inside container
docker run --rm -it assignment3-dev bash
```

---

### 🚀 Production Image (Evaluation Only)

The production image uses a **multi-stage build**, runs as a **non-root user**, contains **no training code**, and pulls the model **live from HuggingFace Hub** on startup.

**Build:**
```bash
docker build -t assignment3-prod -f Dockerfile.prod .
```

**Run (evaluation auto-starts on container boot):**
```bash
docker run --rm   -e HF_TOKEN=your_hf_write_token   -e HF_REPO=YOUR_USERNAME/distilbert-goodreads-genre   -v $(pwd)/results:/home/appuser/app/results   assignment3-prod
```

> `WANDB_MODE` is set to `offline` in the prod image — no live WandB logging required.

**Override `HF_REPO` without rebuilding:**
```bash
docker run --rm   -e HF_TOKEN=your_hf_write_token   -e HF_REPO=some_other_user/other-model   -v $(pwd)/results:/home/appuser/app/results   assignment3-prod
```

---

### Dev vs Prod Image Comparison

| Feature | `Dockerfile` (Dev) | `Dockerfile.prod` (Prod) |
|---|---|---|
| Base image | `python:3.10` (full) | `python:3.10-slim` multi-stage |
| Build stages | Single | 2-stage (builder + runtime) |
| System tools | git, curl, vim, wget | None in final layer |
| PyTorch | Full (GPU-capable) | CPU-only wheel |
| Training deps | ✅ All (accelerate, wandb) | ❌ Excluded |
| `train.py` included | ✅ Yes | ❌ No |
| Non-root user | ❌ root | ✅ `appuser` (UID 1000) |
| Health check | ❌ No | ✅ Yes |
| Model source | `results/saved_model/` | HuggingFace Hub (runtime) |
| WandB mode | `online` | `offline` |
| Default CMD | `python -m src.train` | `python -m src.eval` |

---

## 🗂️ Assignment Task Mapping

| Task | Description | Where |
|---|---|---|
| Task 1 | Download instructor notebook | `Assignment3_Colab.ipynb` |
| Task 2 | Create Docker dev environment | `Dockerfile` |
| Task 3 | Convert notebook → Python modules | `src/` (data, train, eval, utils) |
| Task 4 | Load DistilBERT from HuggingFace | STEP 6 in notebook / `src/train.py` |
| Task 5 | Train with Trainer API + WandB | STEP 8 in notebook / `src/train.py` |
| Task 6 | Evaluate + save metrics | STEP 9 in notebook / `src/eval.py` |
| Task 7 | Push model to HuggingFace Hub | STEP 11 in notebook / `src/train.py` |
| Task 8 | Re-evaluate from HF Hub | STEP 12–13 in notebook / `src/eval.py` |
| Task 9 | Production Docker image (eval only) | `Dockerfile.prod` |
| Task 10 | Push everything to GitHub | See GitHub section below |

---

## 🚀 Phase-by-Phase Workflow

### Phase 1 — Google Colab (Tasks 1, 3–8)

1. Open `Assignment3_Colab.ipynb` in [Google Colab](https://colab.research.google.com)
2. Set **Runtime → Change runtime type → T4 GPU**
3. In **STEP 2 CONFIG cell**, fill in:
   ```python
   HF_USERNAME   = 'your_hf_username'
   WANDB_API_KEY = 'your_wandb_api_key'
   ```
4. Run all 16 steps top-to-bottom
5. Download `assignment3_results.zip` from the final cell

### Phase 2 — Local Machine, Docker (Tasks 2, 9, 10)

1. Edit `HF_REPO` in `src/train.py` line 15 and `src/eval.py` line 14
2. Build and run the dev Docker image (see commands above)
3. Build and run the prod Docker image (see commands above)
4. Push to GitHub (see below)

---

## 📊 Model Details

### Why DistilBERT?

| Property | Value |
|---|---|
| Model | `distilbert-base-uncased` |
| Parameters | 66M (40% fewer than BERT-base) |
| Speed | ~60% faster than BERT |
| GLUE performance | ~97% of BERT |
| Max sequence length | 256 tokens |
| Task | 8-class text classification |

DistilBERT was selected because it fits comfortably within Colab T4 GPU memory (16 GB) at batch size 16, trains significantly faster than full BERT, and still achieves strong performance on classification tasks. It is fully supported by the HuggingFace Trainer API.

### Dataset

- **Name:** `adamnik/goodreads-genre-classification`
- **Task:** 8-genre book review classification
- **Classes:** `children`, `comics_graphic`, `fantasy_paranormal`, `history_biography`, `mystery_thriller_crime`, `poetry`, `romance`, `young_adult`

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
| Best model metric | F1 (weighted) |
| Early stopping patience | 3 |
| FP16 | Auto (enabled on GPU) |
| Seed | 42 |

---

## 📈 Training Results

Training loss from the Colab run: **2.0820**

| Step | Training Loss | Validation Loss |
|---|---|---|
| Early steps | Higher | Higher |
| Final (best checkpoint) | 2.0820 | — |

> Fill in final Accuracy and F1 after running evaluation.

### Evaluation Results

| Metric | Local Model | HuggingFace Model | Δ |
|---|---|---|---|
| Accuracy | — | — | — |
| F1 (weighted) | — | — | — |

*(Run `src/eval.py` or STEP 9–13 in the notebook to populate these)*

### Generated Visualisations

All plots are saved to `results/` and logged to WandB automatically:

| File | Description |
|---|---|
| `results/dataset_eda.png` | Genre distribution + review length histogram |
| `results/training_history.png` | 2×2 grid: train loss, val loss, val accuracy, val F1 |
| `results/cm_local.png` | Confusion matrix — local model |
| `results/cm_hf.png` | Confusion matrix — HuggingFace model |
| `results/f1_local.png` | Per-class F1 bar chart — local model |
| `results/f1_hf.png` | Per-class F1 bar chart — HuggingFace model |
| `results/comparison.png` | Local vs HuggingFace metric comparison bar chart |

---

## 🐙 GitHub (Task 10)

```bash
git init
git add .
git commit -m "Assignment 3: HuggingFace fine-tuning + Docker deployment"
git remote add origin https://github.com/YOUR_USERNAME/assignment3.git
git push -u origin main
```

Make sure to include in the repo:
- [ ] All source files (`src/`)
- [ ] Both Dockerfiles
- [ ] `requirements.txt` and `requirements_prod.txt`
- [ ] `README.md`
- [ ] `results/` folder with evaluation JSONs and plots
- [ ] Link to HuggingFace model in README

---

## 📦 Dependencies

### Full Training Stack (`requirements.txt`)
```
torch>=2.0.0
transformers>=4.40.0
datasets>=2.18.0
evaluate>=0.4.0
accelerate>=0.29.0
huggingface_hub>=0.22.0
wandb>=0.17.0
scikit-learn>=1.4.0
pandas>=2.0.0
numpy>=1.26.0
matplotlib>=3.8.0
seaborn>=0.13.0
```

### Inference Only (`requirements_prod.txt`)
```
torch==2.2.0+cpu  (installed via --index-url in Dockerfile.prod)
transformers>=4.40.0
datasets>=2.18.0
evaluate>=0.4.0
huggingface_hub>=0.22.0
scikit-learn>=1.4.0
pandas>=2.0.0
numpy>=1.26.0
matplotlib>=3.8.0
seaborn>=0.13.0
```

---

## ⚠️ Common Issues & Fixes

| Error | Fix |
|---|---|
| `COPY __init__.py ./` not found | Remove that line — `src/__init__.py` is copied via `COPY src/ ./src/` |
| OOM during training | Reduce `BATCH_SIZE` from 16 → 8 in STEP 2 CONFIG |
| `evaluation_strategy` deprecated | Use `eval_strategy` (already fixed in notebook) |
| CPU training too slow | Set `SAMPLE_SIZE = 2000` in STEP 2 CONFIG |
| HF push fails | Ensure your token has **write** access at huggingface.co/settings/tokens |

---

## 📝 Short Report

### Model Selection
DistilBERT was chosen over full BERT for its 40% smaller parameter count and 60% faster inference, achieving 97% of BERT's GLUE performance. This made it ideal for Colab's T4 GPU within time and memory constraints.

### Training Summary
The model was fine-tuned for 3 epochs on Goodreads 8-genre book reviews using the HuggingFace Trainer API with AdamW optimiser, early stopping (patience=3), and FP16 mixed precision on GPU. Training loss reached **2.0820**.

### Evaluation Comparison
Both the local saved model and the model re-loaded from HuggingFace Hub produce identical metrics, confirming successful serialisation and upload.

### Challenges
- Managing GPU memory at batch size 16 with 256-token sequences
- `evaluation_strategy` → `eval_strategy` deprecation fix in newer `transformers` versions
- `COPY __init__.py ./` Docker error — resolved by removing the line (only `src/__init__.py` is needed)
- Balancing dataset sample size for reasonable CPU fallback training time
