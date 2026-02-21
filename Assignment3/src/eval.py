"""
eval.py  –  Evaluation from LOCAL model + from HUGGING FACE model
             with full visualisations and WandB logging
"""
import os, json, random
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import evaluate
from src.data import get_dataset
from src.utils import (
    GENRES, ID2LABEL, LABEL2ID, NUM_LABELS,
    set_seed, get_device, save_json
)

# ── Config ────────────────────────────────────────────────────
HF_REPO       = "Saumya3007/distilbert-goodreads-genre"   # ← change this
LOCAL_MODEL   = "./results/saved_model"
RESULTS_DIR   = "./results"
WANDB_PROJECT = "hf-docker-assignment"
SAMPLE_SIZE   = None   # match whatever was used in train.py
SEED          = 42
BATCH_SIZE    = 16

os.makedirs(RESULTS_DIR, exist_ok=True)
accuracy_m = evaluate.load("accuracy")
f1_m       = evaluate.load("f1")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_m.compute(predictions=preds, references=labels)["accuracy"],
        "f1":       f1_m.compute(predictions=preds, references=labels, average="weighted")["f1"],
    }


def _make_trainer(model, tokenizer, ds):
    args = TrainingArguments(
        output_dir=RESULTS_DIR,
        per_device_eval_batch_size=BATCH_SIZE,
        report_to="none",
        no_cuda=not torch.cuda.is_available(),
    )
    return Trainer(model=model, args=args, eval_dataset=ds["test"],
                   tokenizer=tokenizer, compute_metrics=compute_metrics)


# ── Confusion-matrix heat-map ─────────────────────────────────
def plot_confusion_matrix(true_labels, pred_labels, title, fname):
    genre_dict = defaultdict(int)
    for t, p in zip(true_labels, pred_labels):
        genre_dict[(ID2LABEL[t], ID2LABEL[p])] += 1
    rows = [{"True Genre": tg, "Predicted Genre": pg, "Count": c}
            for (tg, pg), c in genre_dict.items()]
    df = pd.DataFrame(rows).pivot_table(
        index="True Genre", columns="Predicted Genre", values="Count").fillna(0)

    plt.figure(figsize=(10, 8))
    sns.set(style="ticks", font_scale=1.1)
    sns.heatmap(df, linewidths=1, cmap="Purples", annot=True, fmt=".0f")
    plt.title(title, fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, fname)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")
    return path


# ── Per-class bar-chart ───────────────────────────────────────
def plot_per_class(true_labels, pred_labels, title, fname):
    from sklearn.metrics import classification_report
    report = classification_report(
        [ID2LABEL[i] for i in true_labels],
        [ID2LABEL[i] for i in pred_labels],
        output_dict=True
    )
    genres = GENRES
    f1s = [report.get(g, {}).get("f1-score", 0) for g in genres]

    plt.figure(figsize=(12, 5))
    bars = plt.bar(genres, f1s, color=plt.cm.Purples(np.linspace(0.4, 0.9, len(genres))))
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("F1 Score")
    plt.title(title)
    plt.ylim(0, 1.1)
    for bar, val in zip(bars, f1s):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f"{val:.2f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, fname)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")
    return path


# ── Comparison bar-chart ──────────────────────────────────────
def plot_comparison(local_metrics, hf_metrics, fname="comparison.png"):
    metrics = ["accuracy", "f1"]
    local_vals = [local_metrics[f"eval_{m}"] for m in metrics]
    hf_vals    = [hf_metrics[f"eval_{m}"]    for m in metrics]

    x = np.arange(len(metrics))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    b1 = ax.bar(x - w/2, local_vals, w, label="Local model",        color="#5b2c8d")
    b2 = ax.bar(x + w/2, hf_vals,    w, label="HuggingFace model",  color="#a78ec5")
    ax.set_xticks(x); ax.set_xticklabels([m.capitalize() for m in metrics])
    ax.set_ylim(0, 1.15); ax.set_ylabel("Score")
    ax.set_title("Local vs Hugging Face Model Metrics")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    for b in [b1, b2]:
        for bar in b:
            h = bar.get_height()
            ax.text(bar.get_x()+bar.get_width()/2, h+0.01, f"{h:.3f}",
                    ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, fname)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")
    return path


# ── Sample predictions printout ───────────────────────────────
def show_samples(model, tokenizer, raw_texts, true_labels, pred_labels, n=5):
    device = next(model.parameters()).device
    print("\n── Correct predictions ──────────────────────────────────")
    correct = [(t, p, tx) for t, p, tx in zip(true_labels, pred_labels, raw_texts) if t == p]
    for t, p, tx in random.sample(correct, min(n, len(correct))):
        print(f"  LABEL: {ID2LABEL[t]}")
        print(f"  TEXT : {tx[:120]}…\n")

    print("── Misclassifications ───────────────────────────────────")
    wrong = [(t, p, tx) for t, p, tx in zip(true_labels, pred_labels, raw_texts) if t != p]
    for t, p, tx in random.sample(wrong, min(n, len(wrong))):
        print(f"  TRUE: {ID2LABEL[t]}  |  PREDICTED: {ID2LABEL[p]}")
        print(f"  TEXT: {tx[:120]}…\n")


def evaluate_model(model_path_or_repo: str, ds, tag: str):
    """Load a model (local path OR HF repo), run evaluation, return metrics + preds."""
    print(f"\n  Loading model from: {model_path_or_repo}")
    tokenizer = AutoTokenizer.from_pretrained(model_path_or_repo)
    model     = AutoModelForSequenceClassification.from_pretrained(model_path_or_repo)

    trainer = _make_trainer(model, tokenizer, ds)
    metrics = trainer.evaluate()
    pred_out = trainer.predict(ds["test"])
    preds = np.argmax(pred_out.predictions, axis=-1)
    true  = pred_out.label_ids

    print(f"  [{tag}] accuracy={metrics['eval_accuracy']:.4f}  f1={metrics['eval_f1']:.4f}")
    save_json(metrics, f"{RESULTS_DIR}/{tag}_metrics.json")
    return metrics, preds, true, tokenizer, model


def main():
    set_seed(SEED)
    wandb.init(project=WANDB_PROJECT, name="evaluation-run", config={"mode": "eval"})

    # ── Load data ─────────────────────────────────────────────
    print("[1/4] Loading dataset …")
    ds, _ = get_dataset(sample_size=SAMPLE_SIZE, seed=SEED)

    # Keep raw texts for qualitative analysis
    from src.data import load_raw_dataset
    raw_ds = load_raw_dataset(sample_size=SAMPLE_SIZE, seed=SEED)
    test_texts = raw_ds["test"]["review_text"]
    # align indices after shuffle/select
    if SAMPLE_SIZE:
        test_texts = list(test_texts)[:len(ds["test"])]

    # ── Local evaluation ──────────────────────────────────────
    print("\n[2/4] Evaluating LOCAL model …")
    local_metrics, local_preds, true_labels, *_ = evaluate_model(LOCAL_MODEL, ds, "local")

    cm_path_local = plot_confusion_matrix(true_labels, local_preds,
        "Confusion Matrix — Local Model", "cm_local.png")
    pc_path_local = plot_per_class(true_labels, local_preds,
        "Per-Class F1 — Local Model", "f1_local.png")

    # ── HuggingFace evaluation ────────────────────────────────
    print("\n[3/4] Evaluating HUGGINGFACE model …")
    hf_metrics, hf_preds, _, hf_tok, hf_model = evaluate_model(HF_REPO, ds, "huggingface")

    cm_path_hf = plot_confusion_matrix(true_labels, hf_preds,
        "Confusion Matrix — HuggingFace Model", "cm_hf.png")
    pc_path_hf = plot_per_class(true_labels, hf_preds,
        "Per-Class F1 — HuggingFace Model", "f1_hf.png")

    # ── Comparison ────────────────────────────────────────────
    cmp_path = plot_comparison(local_metrics, hf_metrics)

    # ── Qualitative samples ───────────────────────────────────
    show_samples(hf_model, hf_tok, test_texts, list(true_labels), list(hf_preds))

    # ── WandB logging ─────────────────────────────────────────
    print("\n[4/4] Logging to WandB …")
    wandb.log({
        "local/accuracy":  local_metrics["eval_accuracy"],
        "local/f1":        local_metrics["eval_f1"],
        "hf/accuracy":     hf_metrics["eval_accuracy"],
        "hf/f1":           hf_metrics["eval_f1"],
        "cm_local":        wandb.Image(cm_path_local),
        "cm_hf":           wandb.Image(cm_path_hf),
        "f1_local":        wandb.Image(pc_path_local),
        "f1_hf":           wandb.Image(pc_path_hf),
        "comparison":      wandb.Image(cmp_path),
    })

    # ── Print comparison table ────────────────────────────────
    print("\n══════════════ FINAL COMPARISON ══════════════")
    print(f"{'Metric':<12} {'Local':>10} {'HuggingFace':>14} {'Δ':>8}")
    print("─" * 48)
    for m in ["accuracy", "f1"]:
        lv = local_metrics[f"eval_{m}"]
        hv = hf_metrics[f"eval_{m}"]
        print(f"{m:<12} {lv:>10.4f} {hv:>14.4f} {hv-lv:>+8.4f}")
    print("══════════════════════════════════════════════")

    wandb.finish()


if __name__ == "__main__":
    main()
