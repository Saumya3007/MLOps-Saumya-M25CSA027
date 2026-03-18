import os, torch
MODEL_NAME  = "distilbert-base-uncased"
HF_USERNAME = os.getenv("HF_USERNAME", "Saumya3007")
HF_REPO     = f"{HF_USERNAME}/distilbert-goodreads-genre"
GENRES = ["children","comics_graphic","fantasy_paranormal","history_biography",
          "mystery_thriller_crime","poetry","romance","young_adult"]
LABEL2ID = {g: i for i, g in enumerate(GENRES)}
ID2LABEL = {i: g for i, g in enumerate(GENRES)}
NUM_LABELS = len(GENRES)
MAX_LENGTH = 512; BATCH_SIZE = 16 if torch.cuda.is_available() else 8
EPOCHS = 3; LR = 5e-5; WARMUP_STEPS = 100; WEIGHT_DECAY = 0.01
HEAD_PER_GENRE = 10000; SAMPLE_PER_GENRE = 1000; TRAIN_SPLIT = 0.8; SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("results", exist_ok=True)

