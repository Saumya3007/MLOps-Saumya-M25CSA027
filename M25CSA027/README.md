Q2# CIFAR-10 Image Classification with ResNet18

This repository contains the implementation and evaluation of a **ResNet18** model trained on the CIFAR-10 dataset. The project integrates **Hugging Face** for model hosting and **Weights & Biases (W&B)** for experiment tracking and visualization.

## 📊 Performance Summary
- **Final Test Accuracy:** 87.20%
- **Model Architecture:** ResNet18
- **Framework:** PyTorch

### Class-wise Accuracy (for Exam Sheet)
| Class ID | Class Name | Accuracy |
| :--- | :--- | :--- |
| 0 | Airplane | 85.00% |
| 1 | Automobile | 94.00% |
| 2 | Bird | 73.00% |
| 3 | Cat | 73.00% |
| 4 | Deer | 88.00% |
| 5 | Dog | 85.00% |
| 6 | Frog | 95.00% |
| 7 | Horse | 92.00% |
| 8 | Ship | 91.00% |
| 9 | Truck | 96.00% |

---

## 🚀 Model & Logs

### Hugging Face Hub
The trained model weights are pushed to Hugging Face:
👉 [Saumya3007/cifar10-resnet18](https://huggingface.co/Saumya3007/cifar10-resnet18)

### Weights & Biases (W&B)
You can view the training curves, confusion matrix, and sample predictions here:
- **Project Report Page:** [cifar10_resnet18](https://wandb.ai/pancholisaumya-iit/cifar10_resnet18](https://api.wandb.ai/links/pancholisaumya-iit/kexrb4wb))

---

## 🛠️ How to Load the Model
You can easily load the model directly from Hugging Face for inference:

```python
import torch
from huggingface_hub import hf_hub_download

# Download the model
model_path = hf_hub_download(repo_id="Saumya3007/cifar10-resnet18", filename="pytorch_model.bin")

# Load with PyTorch
model = torch.load(model_path)
model.eval()
print("Model loaded successfully from HuggingFace!")
