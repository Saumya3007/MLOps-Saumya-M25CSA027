import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import glob


NUM_CLASSES = 23
IMG_SIZE = 256
BATCH_SIZE = 8
EPOCHS = 20
LR = 1e-3
SEED = 42
DATA_DIR = "data"
SAVE_DIR = "outputs"

os.makedirs(SAVE_DIR, exist_ok=True)
torch.manual_seed(SEED)
np.random.seed(SEED)


def encode_mask(mask_img):
    mask_arr = np.array(mask_img.convert("RGB"))
    label = np.zeros((mask_arr.shape[0], mask_arr.shape[1]), dtype=np.int64)
    colors = np.array([
        [0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
        [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
        [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
        [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
        [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
        [0, 64, 128], [128, 64, 128], [0, 192, 128]
    ], dtype=np.uint8)
    for idx, color in enumerate(colors):
        match = np.all(mask_arr == color, axis=-1)
        label[match] = idx
    return label


class CityScapeDataset(Dataset):
    def __init__(self, img_paths, mask_paths, img_size=IMG_SIZE, augment=False):
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.img_size = img_size
        self.augment = augment
        self.img_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx])

        if self.augment and np.random.rand() > 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        img_tensor = self.img_transform(img)
        mask_resized = mask.resize((self.img_size, self.img_size), Image.NEAREST)
        label = encode_mask(mask_resized)
        label_tensor = torch.from_numpy(label).long()
        return img_tensor, label_tensor


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, in_ch=3, num_classes=NUM_CLASSES):
        super().__init__()
        self.enc1 = DoubleConv(in_ch, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)
        self.bottleneck = DoubleConv(512, 1024)
        self.pool = nn.MaxPool2d(2)
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = DoubleConv(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = DoubleConv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = DoubleConv(128, 64)
        self.out_conv = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out_conv(d1)


def compute_miou(preds, labels, num_classes=NUM_CLASSES):
    iou_list = []
    preds = preds.view(-1)
    labels = labels.view(-1)
    for cls in range(num_classes):
        pred_c = (preds == cls)
        true_c = (labels == cls)
        inter = (pred_c & true_c).sum().item()
        union = (pred_c | true_c).sum().item()
        if union == 0:
            iou_list.append(float('nan'))
        else:
            iou_list.append(inter / union)
    valid = [v for v in iou_list if not np.isnan(v)]
    return float(np.mean(valid)) if valid else 0.0


def compute_mdice(preds, labels, num_classes=NUM_CLASSES):
    dice_list = []
    preds = preds.view(-1)
    labels = labels.view(-1)
    for cls in range(num_classes):
        pred_c = (preds == cls)
        true_c = (labels == cls)
        inter = (pred_c & true_c).sum().item()
        denom = pred_c.sum().item() + true_c.sum().item()
        if denom == 0:
            dice_list.append(float('nan'))
        else:
            dice_list.append(2 * inter / denom)
    valid = [v for v in dice_list if not np.isnan(v)]
    return float(np.mean(valid)) if valid else 0.0


def load_data():
    rgb_dir = os.path.join(DATA_DIR, "CameraRGB")
    mask_dir = os.path.join(DATA_DIR, "CameraMask")
    img_paths = sorted(glob.glob(os.path.join(rgb_dir, "*")))
    mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*")))
    assert len(img_paths) == len(mask_paths), "Mismatch between images and masks"
    print(f"Total samples: {len(img_paths)}")
    train_imgs, test_imgs, train_masks, test_masks = train_test_split(
        img_paths, mask_paths, test_size=0.2, random_state=SEED
    )
    return train_imgs, test_imgs, train_masks, test_masks


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_imgs, test_imgs, train_masks, test_masks = load_data()

    np.save(os.path.join(SAVE_DIR, "test_imgs.npy"), np.array(test_imgs))
    np.save(os.path.join(SAVE_DIR, "test_masks.npy"), np.array(test_masks))

    train_ds = CityScapeDataset(train_imgs, train_masks, augment=True)
    test_ds = CityScapeDataset(test_imgs, test_masks)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    model = UNet(num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    train_losses, train_mious, train_mdices = [], [], []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        ep_loss, ep_miou, ep_mdice = 0.0, 0.0, 0.0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            preds = logits.argmax(dim=1)
            ep_loss += loss.item()
            ep_miou += compute_miou(preds.cpu(), masks.cpu())
            ep_mdice += compute_mdice(preds.cpu(), masks.cpu())

        n = len(train_loader)
        avg_loss = ep_loss / n
        avg_miou = ep_miou / n
        avg_mdice = ep_mdice / n
        train_losses.append(avg_loss)
        train_mious.append(avg_miou)
        train_mdices.append(avg_mdice)
        scheduler.step(avg_loss)
        print(f"Epoch [{epoch}/{EPOCHS}] Loss={avg_loss:.4f} mIoU={avg_miou:.4f} mDice={avg_mdice:.4f}")

    torch.save(model.state_dict(), os.path.join(SAVE_DIR, "unet_cityscape.pth"))

    model.eval()
    test_miou, test_mdice = 0.0, 0.0
    with torch.no_grad():
        for imgs, masks in test_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            logits = model(imgs)
            preds = logits.argmax(dim=1)
            test_miou += compute_miou(preds.cpu(), masks.cpu())
            test_mdice += compute_mdice(preds.cpu(), masks.cpu())
    test_miou /= len(test_loader)
    test_mdice /= len(test_loader)
    print(f"\nTest mIoU={test_miou:.4f}  Test mDice={test_mdice:.4f}")

    metrics = {
        "train_losses": train_losses,
        "train_mious": train_mious,
        "train_mdices": train_mdices,
        "test_miou": test_miou,
        "test_mdice": test_mdice
    }
    with open(os.path.join(SAVE_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    epochs_range = list(range(1, EPOCHS + 1))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(epochs_range, train_losses, 'b-o', linewidth=2)
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True)

    axes[1].plot(epochs_range, train_mious, 'g-o', linewidth=2)
    axes[1].set_title("Training mIoU")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("mIoU")
    axes[1].grid(True)

    axes[2].plot(epochs_range, train_mdices, 'r-o', linewidth=2)
    axes[2].set_title("Training mDice")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("mDice")
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "training_plots.png"), dpi=150, bbox_inches='tight')
    print(f"Plots saved to {SAVE_DIR}/training_plots.png")
    print(f"Metrics saved to {SAVE_DIR}/metrics.json")
    return test_miou, test_mdice


if __name__ == "__main__":
    test_miou, test_mdice = train()
    print(f"\nFinal Results:")
    print(f"Question2: mIOU: {test_miou:.4f} and mDICE: {test_mdice:.4f}")
