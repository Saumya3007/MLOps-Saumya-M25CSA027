import torch.nn as nn
import timm
from peft import LoraConfig, get_peft_model
from src.config import MODEL_NAME, NUM_CLASSES, LORA_TARGETS


def build_vit_baseline() -> nn.Module:
    """ViT-S with ONLY classification head trainable (no LoRA)."""
    model = timm.create_model(MODEL_NAME, pretrained=True)
    for p in model.parameters():
        p.requires_grad = False
    in_features = model.head.in_features
    model.head = nn.Linear(in_features, NUM_CLASSES)  # head is trainable by default
    return model


from src.config import MODEL_NAME, NUM_CLASSES, LORA_TARGETS

def build_vit_lora(rank: int, alpha: float, dropout: float) -> nn.Module:
    base = timm.create_model(MODEL_NAME, pretrained=True)
    in_features = base.head.in_features
    base.head = nn.Linear(in_features, NUM_CLASSES)

    lora_cfg = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=LORA_TARGETS,   
        bias="none",
    )
    model = get_peft_model(base, lora_cfg)
    for n, p in model.named_parameters():
        if "head" in n:
            p.requires_grad = True
    return model


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_total_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())