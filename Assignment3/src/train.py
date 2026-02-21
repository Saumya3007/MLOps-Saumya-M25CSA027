"""
train.py  –  Fine-tune DistilBERT with Trainer API + WandB logging
"""
import os
import numpy as np
import wandb
from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments, Trainer,
    EarlyStoppingCallback,
)
import evaluate
from huggingface_hub import HfApi
from src.data import get_dataset
from src.utils import (
    MODEL_CHECKPOINT, NUM_LABELS, ID2LABEL, LABEL2ID,
    set_seed, get_device, save_json
)

# ── Config ─────────────────────────────────────────────────────
HF_REPO        = "Saumya3007/distilbert-goodreads-genre"  
WANDB_PROJECT  = "hf-docker-assignment"
OUTPUT_DIR     = "./results/checkpoints"
EPOCHS         = 3
BATCH_SIZE     = 16
LR             = 2e-5
WARMUP_STEPS   = 200
WEIGHT_DECAY   = 0.01
SAMPLE_SIZE    = None   # set e.g. 4000 for CPU / quick test
SEED           = 42


# ── Metrics ────────────────────────────────────────────────────
accuracy_m = evaluate.load("accuracy")
f1_m       = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_m.compute(predictions=preds, references=labels)["accuracy"],
        "f1":       f1_m.compute(predictions=preds, references=labels, average="weighted")["f1"],
    }


def main():
    set_seed(SEED)
    device = get_device()

    # ── Data ──────────────────────────────────────────────────
    print("\n[1/5] Loading & tokenising dataset …")
    ds, tokenizer = get_dataset(sample_size=SAMPLE_SIZE, seed=SEED)

    # ── Model ─────────────────────────────────────────────────
    print(f"\n[2/5] Loading model: {MODEL_CHECKPOINT}")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_CHECKPOINT,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    total = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total:,}")

    # ── WandB ─────────────────────────────────────────────────
    wandb.init(project=WANDB_PROJECT, config={
        "model": MODEL_CHECKPOINT, "epochs": EPOCHS,
        "batch_size": BATCH_SIZE, "lr": LR,
    })

    # ── Training args ─────────────────────────────────────────
    fp16 = (device.type == "cuda")
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LR,
        weight_decay=WEIGHT_DECAY,
        warmup_steps=WARMUP_STEPS,
        evaluation_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_dir="./results/logs",
        logging_steps=50,
        report_to="wandb",
        fp16=fp16,
        seed=SEED,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds["train"],
        eval_dataset=ds["val"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    # ── Train ─────────────────────────────────────────────────
    print("\n[3/5] Training …")
    train_result = trainer.train()
    print(f"  Training loss : {train_result.training_loss:.4f}")

    # ── Save locally ──────────────────────────────────────────
    print("\n[4/5] Saving model locally …")
    LOCAL_MODEL_DIR = "./results/saved_model"
    trainer.save_model(LOCAL_MODEL_DIR)
    tokenizer.save_pretrained(LOCAL_MODEL_DIR)
    print(f"  Saved to {LOCAL_MODEL_DIR}")

    # ── Push to HF Hub ────────────────────────────────────────
    print(f"\n[5/5] Pushing model to HuggingFace: {HF_REPO}")
    model.push_to_hub(HF_REPO)
    tokenizer.push_to_hub(HF_REPO)
    print(f"  ✅ Model live at: https://huggingface.co/{HF_REPO}")

    save_json(train_result.metrics, "results/train_metrics.json")
    wandb.finish()
    return trainer, ds, tokenizer


if __name__ == "__main__":
    main()
