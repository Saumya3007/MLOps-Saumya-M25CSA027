# Assignment 5 — Q2: Adversarial Attacks with IBM ART

## 🔗 Links
| Resource | Link |
|---|---|
| 📊 WandB Report | [View Report](https://wandb.ai/pancholisaumya-iit/assignment5-adversarial/reports/Q2_Assignment5--VmlldzoxNjQyMTUyMw) |
| 🤗 HuggingFace Models | [Saumya3007/assignment5_q2-adversarial](https://huggingface.co/Saumya3007/assignment5_q2-adversarial) |

## 📋 Results Summary
| Condition | Accuracy |
|---|---|
| Clean (ResNet-18) | **88.35%** ✅ |
| FGSM Scratch | 69.51% |
| FGSM IBM ART | 32.10% |
| PGD IBM ART | 14.00% |
| BIM IBM ART | 14.00% |
| Detector PGD (ResNet-34) | **70.83%** ✅ |
| Detector BIM (ResNet-34) | **70.87%** ✅ |

## 🗂 Project Structure
Q2/
├── main.py # Full pipeline entry point
├── .env # WandB / HuggingFace credentials
├── requirements.txt
├── models/resnet_models.py # ResNet-18 (victim) & ResNet-34 (detector)
├── attacks/
│ ├── fgsm_scratch.py # FGSM without ART
│ └── art_attacks.py # FGSM, PGD, BIM via IBM ART
├── detectors/
│ └── train_detector.py # Binary adversarial detector (ResNet-34)
└── utils/
├── data_loader.py
├── trainer.py
└── visualize.py

text

## ⚙️ Setup & Run
```bash
pip install -r requirements.txt

# Fill .env
WANDB_API_KEY=your_key
WANDB_PROJECT=assignment5-adversarial
HF_TOKEN=your_token
HF_REPO_ID=your_username/assignment5_q2-adversarial

# Run full pipeline
python main.py

# Skip training if checkpoint exists
python main.py --skip_train
```

## 🛠 Tech Stack
- PyTorch + torchvision (ResNet-18, ResNet-34)
- IBM Adversarial Robustness Toolbox (ART ≥1.17)
- Weights & Biases (experiment tracking)
- HuggingFace Hub (model hosting)
- CIFAR-10 dataset