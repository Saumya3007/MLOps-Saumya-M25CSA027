"""
data.py  –  Dataset loading & tokenisation
Dataset: adamnik/goodreads-genre-classification (Goodreads book-reviews, 8 genres)
"""
import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoTokenizer
from src.utils import MODEL_CHECKPOINT, LABEL2ID, GENRES, set_seed

MAX_LENGTH = 256   # keep memory low; raise to 512 if GPU allows


def load_raw_dataset(sample_size: int = None, seed: int = 42):
    """
    Load the Goodreads 8-genre dataset from HuggingFace.
    Falls back to a tiny synthetic set for smoke-testing on CPU.
    """
    set_seed(seed)
    try:
        print("Loading dataset from HuggingFace Hub …")
        # This dataset has columns: review_text, genre
        ds = load_dataset("adamnik/goodreads-genre-classification")
        print(f"  train={len(ds['train'])}  test={len(ds['test'])}")
    except Exception as e:
        print(f"[WARN] Could not load HF dataset ({e}). Using tiny synthetic fallback.")
        texts = [f"sample review text number {i}" for i in range(200)]
        labels = [GENRES[i % len(GENRES)] for i in range(200)]
        split = int(0.8 * len(texts))
        ds = DatasetDict({
            "train": Dataset.from_dict({"review_text": texts[:split],  "genre": labels[:split]}),
            "test":  Dataset.from_dict({"review_text": texts[split:], "genre": labels[split:]}),
        })

    # Optional down-sampling (faster iteration / CPU training)
    if sample_size:
        ds["train"] = ds["train"].shuffle(seed=seed).select(range(min(sample_size, len(ds["train"]))))
        ds["test"]  = ds["test"].shuffle(seed=seed).select(range(min(sample_size // 5, len(ds["test"]))))

    return ds


def preprocess_dataset(ds, tokenizer=None, seed: int = 42):
    """Add integer label column, tokenise, train/val split."""
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

    def add_label_id(example):
        example["label"] = LABEL2ID[example["genre"]]
        return example

    def tokenise(batch):
        return tokenizer(
            batch["review_text"],
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
        )

    ds = ds.map(add_label_id)
    ds = ds.map(tokenise, batched=True, batch_size=256)

    # Create validation split from train
    split = ds["train"].train_test_split(test_size=0.1, seed=seed)
    ds = DatasetDict({
        "train": split["train"],
        "val":   split["test"],
        "test":  ds["test"],
    })

    keep_cols = ["input_ids", "attention_mask", "label"]
    for key in ds:
        remove = [c for c in ds[key].column_names if c not in keep_cols]
        ds[key] = ds[key].remove_columns(remove)
        ds[key].set_format("torch")

    print(f"  train={len(ds['train'])}  val={len(ds['val'])}  test={len(ds['test'])}")
    return ds, tokenizer


def get_dataset(sample_size=None, seed=42):
    raw = load_raw_dataset(sample_size=sample_size, seed=seed)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
    ds, tokenizer = preprocess_dataset(raw, tokenizer, seed=seed)
    return ds, tokenizer
