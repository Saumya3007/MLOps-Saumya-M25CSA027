# DistilBERT Goodreads Genre Classifier
### Assignment 3 — End-to-End Hugging Face Model Training & Docker Deployment
**Author:** Saumya Pancholi | IIT Jodhpur | ML/DL Operations

---

## 📌 Overview
Fine-tuned [`distilbert-base-uncased`](https://huggingface.co/distilbert-base-uncased) on 8,000 Goodreads book reviews (UCSD Book Graph) to classify reviews into 8 literary genres. Trained with the HuggingFace Trainer API, tracked with WandB, containerised with Docker, and published to HuggingFace Hub.

---

## 🔗 Links
| Resource | URL |
|----------|-----|
| 🤗 HuggingFace Model | https://huggingface.co/Saumya3007/distilbert-goodreads-genre |
| 📊 WandB  | https://api.wandb.ai/links/pancholisaumya-iit/nnge4fom |
| 📓 Colab Notebook | https://colab.research.google.com/drive/1XYUzMH7J52fsw5dhgSdw_XzOf-9cqCw5?usp=sharing |

---

## 📂 Project Structure
```
.
├── src/
│   ├── __init__.py
│   ├── utils.py        # Config constants, DEVICE, LABEL mappings
│   ├── data.py         # Data loading from UCSD URLs, Dataset class
│   ├── train.py        # Training pipeline (Trainer API)
│   └── eval.py         # Evaluation, confusion heatmap, metric saving
├── Dockerfile           # Dev image: train + eval
├── Dockerfile.prod      # Prod image: eval-only, pulls from HF Hub
├── requirements.txt
├── Assignment3_Colab.ipynb
└── README.md
```

---

## 🗂️ Dataset
- **Source:** [UCSD Goodreads Book Graph](https://mengtingwan.github.io/data/goodreads.html)
- **Genres (8):** children, comics_graphic, fantasy_paranormal, history_biography, mystery_thriller_crime, poetry, romance, young_adult
- **Size:** 1,000 reviews/genre → **6,400 train / 1,600 test** (80/20 split)
- **Loading:** Streamed directly from UCSD `.json.gz` URLs (no manual download needed)

---

## 🧠 Model Selection — Why DistilBERT?
| Property | DistilBERT | BERT-base |
|----------|-----------|-----------|
| Parameters | 66.9M | 110M |
| Inference speed | 60% faster | baseline |
| GLUE score retention | ~97% | 100% |
| Colab T4 (16 GB) at batch 16 | ✅ Fits | ⚠️ Tight |

DistilBERT provides the best trade-off between accuracy and compute efficiency for this assignment's constraints.

---

## ⚙️ Training Configuration
| Hyperparameter | Value |
|----------------|-------|
| Model | distilbert-base-uncased |
| Max token length | 512 |
| Batch size | 16 |
| Epochs | 3 |
| Learning rate | 5e-5 |
| Warmup steps | 100 |
| Weight decay | 0.01 |
| FP16 | ✅ (CUDA) |
| Early stopping patience | 3 |

---

## 📈 Results

### Training Progress
| Step | Train Loss | Val Loss | Accuracy |
|------|-----------|----------|----------|
| 100  | 1.9415    | 1.5485   | 0.5013   |
| 400  | 1.2468    | 1.1869   | 0.5944   |
| 800  | 0.8918    | 1.1482   | 0.6125   |
| 1200 | 0.6075    | 1.1858   | 0.6150   |

### Final Evaluation (Test Set — 1,600 samples)
| Metric | Local Model | HF Hub Model | Delta |
|--------|------------|-------------|-------|
| Accuracy | **0.6156** | **0.6156** | 0.0000 |
| Loss | 1.2028 | 1.2028 | 0.0000 |

### Per-Genre Classification Report
| Genre | Precision | Recall | F1 |
|-------|----------|--------|----|
| children | 0.69 | 0.74 | 0.71 |
| comics_graphic | **0.86** | 0.76 | **0.81** |
| fantasy_paranormal | 0.43 | 0.53 | 0.47 |
| history_biography | 0.61 | 0.61 | 0.61 |
| mystery_thriller_crime | 0.61 | 0.57 | 0.59 |
| poetry | 0.77 | **0.79** | 0.78 |
| romance | 0.63 | 0.56 | 0.59 |
| young_adult | 0.37 | 0.36 | 0.37 |
| **macro avg** | **0.62** | **0.62** | **0.62** |

### Baseline Comparison
| Model | Accuracy |
|-------|----------|
| TF-IDF + Logistic Regression | 0.56 |
| **DistilBERT (fine-tuned)** | **0.62** |

---

## 🐳 Docker Usage

### Task 2 — Dev Image (Train + Eval)
```bash
# Build
docker build -t assignment3-dev -f Dockerfile .

# Run training
docker run --rm \
  -e HF_TOKEN=<your_hf_token> \
  -e WANDB_API_KEY=<your_wandb_key> \
  -v $(pwd)/results:/app/results \
  assignment3-dev python -m src.train

# Run evaluation
docker run --rm \
  -e HF_TOKEN=<your_hf_token> \
  -v $(pwd)/results:/app/results \
  assignment3-dev python -m src.eval
```

### Task 9 — Prod Image (Eval Only — pulls from HF Hub)
```bash
# Build
docker build -t assignment3-prod -f Dockerfile.prod .

# Run (auto-evaluates on startup)
docker run --rm \
  -e HF_TOKEN=<your_hf_token> \
  -e HF_REPO=Saumya3007/distilbert-goodreads-genre \
  -v $(pwd)/results:/home/appuser/app/results \
  assignment3-prod
```

