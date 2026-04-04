import os
from dotenv import load_dotenv

load_dotenv()

HF_USERNAME   = os.getenv("HF_USERNAME", "your_hf_username")
HF_TOKEN      = os.getenv("HF_TOKEN", "")
WANDB_KEY     = os.getenv("WANDB_API_KEY", "")
WANDB_PROJECT = os.getenv("WANDB_PROJECT", "vit-cifar100-lora")
WANDB_ENTITY = os.getenv("WANDB_ENTITY") or None

NUM_CLASSES  = 100
MODEL_NAME   = "vit_small_patch16_224.augreg_in21k_ft_in1k"
IMG_SIZE     = 224
BATCH_SIZE   = 248
NUM_WORKERS  = 4
NUM_EPOCHS   = 10

LR           = 1e-3
WEIGHT_DECAY = 1e-4
DEVICE       = "cuda"

LORA_RANKS   = [2, 4, 8]
LORA_ALPHAS  = [2, 4, 8]
LORA_DROPOUT = 0.1
LORA_TARGETS = ["qkv"]
target_modules=LORA_TARGETS

OPTUNA_TRIALS = 10

ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS_DIR = os.path.join(ROOT, "weights")
RESULTS_DIR = os.path.join(ROOT, "results")
PLOTS_DIR   = os.path.join(ROOT, "plots")