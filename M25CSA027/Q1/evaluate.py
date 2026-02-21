import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import numpy as np

import random
import os
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

from PIL import Image

# ==========================
# Config
# ==========================
DATA_DIR = "data/test/"
MODEL_PATH = "setB.pth"
BATCH_SIZE = 32
NUM_CLASSES = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================
# Transforms
# ==========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ==========================
# Dataset
# ==========================
dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

class_names = dataset.classes
print("Classes:", class_names)

# ==========================
# Load Model
# ==========================
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model = model.to(DEVICE)
model.eval()

print("Model Loaded Successfully!")

# ==========================
# Evaluation
# ==========================
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in dataloader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# ==========================
# Overall Accuracy
# ==========================
overall_acc = accuracy_score(all_labels, all_preds)
print(f"\nOverall Accuracy: {overall_acc * 100:.2f}%")

# ==========================
# F1 Score
# ==========================
macro_f1 = f1_score(all_labels, all_preds, average='macro')
print(f"F1 Score: {macro_f1:.4f}")

print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=class_names))

# ==========================
# Confusion Matrix + Classwise Accuracy
# ==========================
cm = confusion_matrix(all_labels, all_preds)
totals_per_class = cm.sum(axis=1)
per_class_acc = np.zeros(len(class_names), dtype=float)
for i in range(len(class_names)):
    per_class_acc[i] = (cm[i, i] / totals_per_class[i]) if totals_per_class[i] > 0 else 0.0

print("\nClasswise Accuracy:")
for i, cname in enumerate(class_names):
    print(f"{cname}: {per_class_acc[i]*100:.2f}% ({int(cm[i,i])}/{int(totals_per_class[i])})")

import matplotlib.pyplot as plt

def plot_confusion_and_class_accuracy(cm, class_names, per_class_acc):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7), gridspec_kw={'width_ratios': [3, 1]})

    im = ax1.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax1.set_title('Confusion Matrix (counts)')
    tick_marks = np.arange(len(class_names))
    ax1.set_xticks(tick_marks)
    ax1.set_yticks(tick_marks)
    ax1.set_xticklabels(class_names, rotation=45, ha='right')
    ax1.set_yticklabels(class_names)
    thresh = cm.max() / 2. if cm.max() != 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax1.text(j, i, format(int(cm[i, j]), 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    ax1.set_ylabel('True label')
    ax1.set_xlabel('Predicted label')
    fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)

    # Classwise accuracy bar chart
    ax2.barh(np.arange(len(class_names)), per_class_acc * 100, color='skyblue')
    ax2.set_xlim(0, 100)
    ax2.set_yticks(np.arange(len(class_names)))
    ax2.set_yticklabels(class_names)
    ax2.set_xlabel('Accuracy (%)')
    ax2.set_title('Classwise Accuracy')
    for i, v in enumerate(per_class_acc * 100):
        ax2.text(v + 1, i, f"{v:.1f}%", va='center')

    plt.tight_layout()
    plt.show()

# Plot
plot_confusion_and_class_accuracy(cm, class_names, per_class_acc)


def predict_single_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(image)
        probs = torch.softmax(output, dim=1)
        confidence, pred = torch.max(probs, 1)

    print(f"\nImage: {image_path}")
    print(f"Predicted Class: {class_names[pred.item()]}")
    print(f"Confidence: {confidence.item()*100:.2f}%")

# Pick random image from dataset
random_image_path, _ = random.choice(dataset.samples)
predict_single_image(random_image_path)