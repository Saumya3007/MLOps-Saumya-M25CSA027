"""
Two-stream adversarial detector.
Stream A : 512-d frozen ResNet-18 backbone features
Stream B : 6 pixel-stat features [mean, std, min, max, l2, linf]
Head     : 4-layer MLP with BN + GELU + Dropout
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import wandb

MEAN = [0.4914, 0.4822, 0.4465]
STD  = [0.2023, 0.1994, 0.2010]


class DetectorModel(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        for p in self.backbone.parameters():
            p.requires_grad = False

        self.head = nn.Sequential(
            nn.Linear(512 + 6, 1024),
            nn.BatchNorm1d(1024), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),  nn.GELU(), nn.Dropout(0.25),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),  nn.GELU(), nn.Dropout(0.2),
            nn.Linear(256, 2),
        )

    def _pixel_stats(self, x):
        b    = x.size(0)
        flat = x.view(b, -1)
        mu   = flat.mean(1, keepdim=True)
        sd   = flat.std(1,  keepdim=True)
        mn   = flat.min(1,  keepdim=True).values
        mx   = flat.max(1,  keepdim=True).values
        l2   = flat.norm(dim=1, keepdim=True) / flat.size(1) ** 0.5
        linf = (flat - mu).abs().max(1, keepdim=True).values
        return torch.cat([mu, sd, mn, mx, l2, linf], dim=1)  # (B, 6)

    def forward(self, x):
        with torch.no_grad():
            feat = self.backbone(x).view(x.size(0), -1)  # (B, 512)
        stats = self._pixel_stats(x)                      # (B, 6)
        return self.head(torch.cat([feat, stats], dim=1))


class DetectionDS(Dataset):
    def __init__(self, images, labels, augment=False):
        self.images  = images.astype(np.float32)
        self.labels  = labels.astype(np.int64)
        self.norm    = transforms.Normalize(mean=MEAN, std=STD)
        self.augment = augment
        self.flip    = transforms.RandomHorizontalFlip(0.5)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = torch.from_numpy(self.images[idx])
        if self.augment and self.labels[idx] == 0:
            img = self.flip(img)
        return self.norm(img), torch.tensor(self.labels[idx])


def _eval(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with autocast():
                out = model(x)
            correct += out.argmax(1).eq(y).sum().item()
            total   += y.size(0)
    return 100.0 * correct / total


def build_and_train_detector(attack_name, clean_images, adv_images,
                              backbone, device='cuda',
                              save_dir='./checkpoints', epochs=50):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'best_detector_{attack_name}.pth')

    X = np.concatenate([clean_images, adv_images], 0)
    y = np.array([0] * len(clean_images) + [1] * len(adv_images), np.int64)

    rng  = np.random.default_rng(42)
    perm = rng.permutation(len(X))
    X, y = X[perm], y[perm]

    n_val   = max(200, int(0.15 * len(X)))
    n_train = len(X) - n_val

    tr_ds = DetectionDS(X[:n_train], y[:n_train], augment=True)
    va_ds = DetectionDS(X[n_train:], y[n_train:], augment=False)

    tr_ld = DataLoader(tr_ds, batch_size=512, shuffle=True,
                       num_workers=4, pin_memory=True)
    va_ld = DataLoader(va_ds, batch_size=512, shuffle=False,
                       num_workers=4, pin_memory=True)

    model     = DetectorModel(backbone).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    params    = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(params, lr=1e-3, weight_decay=1e-3)
    scaler    = GradScaler()

    def lr_fn(ep):
        if ep < 5:
            return (ep + 1) / 5.0
        return 0.5 * (1 + np.cos(np.pi * (ep - 5) / max(1, epochs - 5)))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_fn)
    best_acc, best_ep = 0.0, 0

    for epoch in range(1, epochs + 1):
        model.train()
        correct, total = 0, 0
        for x, y_b in tr_ld:
            x   = x.to(device, non_blocking=True)
            y_b = y_b.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast():
                out  = model(x)
                loss = criterion(out, y_b)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(optimizer)
            scaler.update()
            correct += out.detach().argmax(1).eq(y_b).sum().item()
            total   += y_b.size(0)

        tr_acc = 100.0 * correct / total
        va_acc = _eval(model, va_ld, device)
        scheduler.step()

        wandb.log({
            f'detector_{attack_name}/epoch'    : epoch,
            f'detector_{attack_name}/train_acc': tr_acc,
            f'detector_{attack_name}/val_acc'  : va_acc,
            f'detector_{attack_name}/lr'       : optimizer.param_groups[0]['lr'],
        })

        if va_acc > best_acc:
            best_acc, best_ep = va_acc, epoch
            torch.save(model.state_dict(), save_path)

        if epoch % 10 == 0 or epoch == epochs:
            print(f'[detector_{attack_name}] ep {epoch:3d}/{epochs} '
                  f'tr={tr_acc:.1f}% val={va_acc:.1f}% best={best_acc:.1f}%')

    model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
    final = _eval(model, va_ld, device)
    wandb.log({f'detector_{attack_name}/final_val_acc': final})
    print(f'\n✓ Detector {attack_name} — best {best_acc:.2f}% @ ep{best_ep} | final {final:.2f}%')
    return model, final