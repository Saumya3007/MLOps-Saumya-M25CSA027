import os, json, random
import numpy as np
import torch

GENRES = [
    "children", "comics_graphic", "fantasy_paranormal",
    "history_biography", "mystery_thriller_crime",
    "poetry", "romance", "young_adult"
]
LABEL2ID = {g: i for i, g in enumerate(GENRES)}
ID2LABEL = {i: g for i, g in enumerate(GENRES)}
NUM_LABELS = len(GENRES)
MODEL_CHECKPOINT = "distilbert-base-uncased"

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[Device] Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("[Device] GPU not available — using CPU")
    return device

def save_json(data, path):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved: {path}")

def load_json(path):
    with open(path) as f:
        return json.load(f)
