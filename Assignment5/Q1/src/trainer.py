import os, time, json
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb
from tqdm import tqdm
from src.config import (LR, WEIGHT_DECAY, NUM_EPOCHS, DEVICE,
                        WEIGHTS_DIR, RESULTS_DIR, WANDB_PROJECT, WANDB_ENTITY)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in tqdm(loader, desc="Train", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += imgs.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in tqdm(loader, desc="Val  ", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss   = criterion(logits, labels)
        total_loss += loss.item() * imgs.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += imgs.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate_classwise(model, loader, num_classes, device):
    """Returns (overall_acc float, per_class_acc list[float])."""
    model.eval()
    class_correct = [0] * num_classes
    class_total   = [0] * num_classes
    for imgs, labels in tqdm(loader, desc="Test ", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        preds = model(imgs).argmax(1)
        for p, l in zip(preds.tolist(), labels.tolist()):
            class_total[l]   += 1
            class_correct[l] += int(p == l)
    per_class_acc = [class_correct[i] / max(class_total[i], 1)
                     for i in range(num_classes)]
    overall_acc   = sum(class_correct) / max(sum(class_total), 1)
    return overall_acc, per_class_acc


def run_experiment(
    model, train_loader, val_loader, experiment_name: str,
    rank=None, alpha=None, dropout=None, use_lora=False,
    epochs=NUM_EPOCHS
):
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    device    = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    model     = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                      lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    run_cfg = dict(experiment=experiment_name, rank=rank, alpha=alpha,
                   dropout=dropout, use_lora=use_lora, epochs=epochs,
                   lr=LR, weight_decay=WEIGHT_DECAY)
    # Replace the wandb.init() call with this:
    wandb_run = wandb.init(
            project=WANDB_PROJECT,
            **({"entity": WANDB_ENTITY} if WANDB_ENTITY else {}),
            name=experiment_name,
            config=run_cfg,
            reinit=True
    )
    # LoRA gradient norm hooks
    lora_grad_norms = {}
    if use_lora:
        def _make_hook(name):
            def hook(grad):
                lora_grad_norms[name] = grad.norm().item()
            return hook
        for n, p in model.named_parameters():
            if ("lora_A" in n or "lora_B" in n) and p.requires_grad:
                p.register_hook(_make_hook(n))

    history, best_val_acc, best_epoch = [], 0.0, 0
    best_ckpt = os.path.join(WEIGHTS_DIR, f"{experiment_name}_best.pt")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc   = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        row = dict(epoch=epoch,
                   train_loss=round(tr_loss,   4),
                   val_loss=round(val_loss,     4),
                   train_acc=round(tr_acc * 100, 2),
                   val_acc=round(val_acc * 100,   2))
        history.append(row)
        print(f"[{experiment_name}] Epoch {epoch:02d}/{epochs} | "
              f"TR Loss {tr_loss:.4f} Acc {tr_acc*100:.2f}% | "
              f"VA Loss {val_loss:.4f} Acc {val_acc*100:.2f}% | "
              f"{time.time()-t0:.1f}s")

        log_dict = {"train/loss": tr_loss, "train/acc": tr_acc,
                    "val/loss":   val_loss, "val/acc":  val_acc,
                    "epoch": epoch}
        for k, v in lora_grad_norms.items():
            log_dict[f"grad_norm/{k}"] = v
        wandb.log(log_dict)

        # ── Save ONLY the single best checkpoint (delete old on improvement) ─
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch   = epoch
            if os.path.exists(best_ckpt):
                os.remove(best_ckpt)          # remove previous best
            torch.save({
                "epoch":      epoch,
                "val_acc":    val_acc,
                "state_dict": model.state_dict(),
                "config":     run_cfg,
            }, best_ckpt)
            print(f"  → New best saved: {best_ckpt}  (val={val_acc*100:.2f}%)")

    wandb_run.finish()

    with open(os.path.join(RESULTS_DIR, f"{experiment_name}_history.json"), "w") as f:
        json.dump({"experiment": experiment_name, "history": history,
                   "best_epoch": best_epoch, "best_val_acc": best_val_acc}, f, indent=2)
    return history, best_val_acc, device