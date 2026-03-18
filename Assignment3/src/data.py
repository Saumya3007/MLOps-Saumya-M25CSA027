import json, gzip, random, requests, torch
from transformers import DistilBertTokenizerFast
from src.utils import *

GENRE_URLS = {
    "poetry":                 "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_poetry.json.gz",
    "children":               "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_children.json.gz",
    "comics_graphic":         "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_comics_graphic.json.gz",
    "fantasy_paranormal":     "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_fantasy_paranormal.json.gz",
    "history_biography":      "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_history_biography.json.gz",
    "mystery_thriller_crime": "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_mystery_thriller_crime.json.gz",
    "romance":                "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_romance.json.gz",
    "young_adult":            "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_young_adult.json.gz",
}

def load_reviews(url, head=HEAD_PER_GENRE, sample_size=SAMPLE_PER_GENRE):
    reviews, count = [], 0
    response = requests.get(url, stream=True)
    with gzip.open(response.raw, "rt", encoding="utf-8") as f:
        for line in f:
            reviews.append(json.loads(line)["review_text"])
            count += 1
            if head and count >= head: break
    return random.sample(reviews, min(sample_size, len(reviews)))

class GoodreadsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings; self.labels = labels
    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx]); return item
    def __len__(self): return len(self.labels)

def load_data():
    random.seed(SEED)
    genre_reviews = {}
    for genre, url in GENRE_URLS.items():
        genre_reviews[genre] = load_reviews(url)
    train_texts, train_labels, test_texts, test_labels = [], [], [], []
    for genre, reviews in genre_reviews.items():
        cutoff = int(len(reviews) * TRAIN_SPLIT)
        train_texts += reviews[:cutoff]; train_labels += [genre] * cutoff
        test_texts  += reviews[cutoff:]; test_labels  += [genre] * (len(reviews) - cutoff)
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
    train_enc = tokenizer(train_texts, truncation=True, padding=True, max_length=MAX_LENGTH)
    test_enc  = tokenizer(test_texts,  truncation=True, padding=True, max_length=MAX_LENGTH)
    return (GoodreadsDataset(train_enc, [LABEL2ID[y] for y in train_labels]),
            GoodreadsDataset(test_enc,  [LABEL2ID[y] for y in test_labels]),
            tokenizer, train_labels, test_labels)

if __name__ == "__main__":
    tr, te, _, _, _ = load_data()
    print(f"train={len(tr)}  test={len(te)}")