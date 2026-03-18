import os, numpy as np, torch, wandb
from sklearn.metrics import accuracy_score
from transformers import (DistilBertForSequenceClassification, TrainingArguments,
                           Trainer, EarlyStoppingCallback)
from src.utils import *
from src.data import load_data

def main():
    train_ds, test_ds, tokenizer, _, _ = load_data()
    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=NUM_LABELS).to(DEVICE)
    wandb.init(project="hf-docker-assignment", name="distilbert-train")
    def compute_metrics(ep):
        return {"accuracy": accuracy_score(ep[1], np.argmax(ep[0], axis=-1))}
    args = TrainingArguments(
        output_dir="results/checkpoints", num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE, per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LR, weight_decay=WEIGHT_DECAY, warmup_steps=WARMUP_STEPS,
        eval_strategy="steps", eval_steps=100, save_strategy="steps", save_steps=100,
        save_total_limit=2, load_best_model_at_end=True,
        metric_for_best_model="accuracy", greater_is_better=True,
        logging_dir="results/logs", logging_steps=100, report_to="wandb",
        fp16=torch.cuda.is_available(), seed=SEED,
    )
    trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=test_ds,
                      compute_metrics=compute_metrics,
                      callbacks=[EarlyStoppingCallback(early_stopping_patience=3)])
    result = trainer.train()
    print(f"Loss: {result.training_loss:.4f}")
    trainer.save_model("results/saved_model")
    tokenizer.save_pretrained("results/saved_model")
    if os.getenv("HF_TOKEN"):
        model.push_to_hub(HF_REPO); tokenizer.push_to_hub(HF_REPO)
    wandb.finish()

if __name__ == "__main__": main()