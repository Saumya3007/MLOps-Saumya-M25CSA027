import os, json, numpy as np, torch, wandb, matplotlib.pyplot as plt, seaborn as sns, pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from transformers import DistilBertForSequenceClassification, TrainingArguments, Trainer
from collections import defaultdict
from src.utils import *
from src.data import load_data

def run_eval(model, test_ds, test_labels, tag):
    trainer = Trainer(model=model,
                      args=TrainingArguments(output_dir=f"results/{tag}_eval",
                                             report_to="none",
                                             per_device_eval_batch_size=BATCH_SIZE))
    metrics  = trainer.evaluate(test_ds)
    pred_out = trainer.predict(test_ds)
    preds    = [ID2LABEL[i] for i in pred_out.predictions.argmax(-1).flatten()]
    print(f"[{tag}]  acc={metrics.get('eval_accuracy',0):.4f}")
    print(classification_report(test_labels, preds))
    genre_cls = defaultdict(int)
    for t, p in zip(test_labels, preds): genre_cls[(t, p)] += 1
    df = (pd.DataFrame([{"True": t, "Pred": p, "N": c} for (t, p), c in genre_cls.items()])
          .pivot_table(index="True", columns="Pred", values="N", fill_value=0))
    plt.figure(figsize=(10, 8))
    sns.heatmap(df, annot=True, fmt="d", cmap="Purples")
    plt.title(f"Confusion — {tag}"); plt.tight_layout()
    plt.savefig(f"results/cm_{tag}.png", dpi=150); plt.close()
    return metrics

def main():
    _, test_ds, _, _, test_labels = load_data()
    wandb.init(project="hf-docker-assignment", name="eval", mode="offline")
    local = DistilBertForSequenceClassification.from_pretrained("results/saved_model").to(DEVICE)
    lm = run_eval(local, test_ds, test_labels, "local")
    with open("results/local_metrics.json", "w") as f: json.dump(lm, f, indent=2)
    hf = DistilBertForSequenceClassification.from_pretrained(
        os.getenv("HF_REPO", HF_REPO)).to(DEVICE)
    hm = run_eval(hf, test_ds, test_labels, "hf_hub")
    with open("results/hf_metrics.json", "w") as f: json.dump(hm, f, indent=2)
    wandb.finish()

if __name__ == "__main__": main()