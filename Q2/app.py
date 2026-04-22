import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


NUM_CLASSES = 23
IMG_SIZE = 256
SAVE_DIR = "outputs"

CLASS_NAMES = [
    "Unlabeled", "Building", "Fence", "Other", "Pedestrian",
    "Pole", "RoadLine", "Road", "Sidewalk", "Vegetation",
    "Vehicles", "Wall", "TrafficSign", "Sky", "Ground",
    "Bridge", "RailTrack", "GuardRail", "TrafficLight", "Static",
    "Dynamic", "Water", "Terrain"
]

COLORS = np.array([
    [0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
    [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
    [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
    [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
    [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
    [0, 64, 128], [128, 64, 128], [0, 192, 128]
], dtype=np.uint8)


def encode_mask(mask_img):
    mask_arr = np.array(mask_img.convert("RGB"))
    label = np.zeros((mask_arr.shape[0], mask_arr.shape[1]), dtype=np.int64)
    for idx, color in enumerate(COLORS):
        match = np.all(mask_arr == color, axis=-1)
        label[match] = idx
    return label


def label_to_color(label_map):
    h, w = label_map.shape
    color_img = np.zeros((h, w, 3), dtype=np.uint8)
    for idx, color in enumerate(COLORS):
        color_img[label_map == idx] = color
    return color_img


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


@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(num_classes=NUM_CLASSES).to(device)
    model_path = os.path.join(SAVE_DIR, "unet_cityscape.pth")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, device


@st.cache_data
def load_metrics():
    metrics_path = os.path.join(SAVE_DIR, "metrics.json")
    with open(metrics_path, "r") as f:
        return json.load(f)


def predict(model, device, img_pil):
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img_tensor = transform(img_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(img_tensor)
        pred = logits.argmax(dim=1).squeeze(0).cpu().numpy()
    return pred


def main():
    st.set_page_config(
        page_title="CityScape Segmentation",
        page_icon="🏙️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.markdown("""
        <style>
        .main { background-color: #0e1117; }
        .stApp { background: linear-gradient(135deg, #0e1117 0%, #1a1f2e 100%); }
        h1 { color: #00d4ff; font-family: 'Segoe UI', sans-serif; }
        h2 { color: #7eb8ff; font-family: 'Segoe UI', sans-serif; }
        h3 { color: #a0c4ff; }
        .metric-card {
            background: linear-gradient(135deg, #1e2a3a, #0d1b2a);
            border: 1px solid #00d4ff44;
            border-radius: 12px;
            padding: 16px;
            text-align: center;
            margin: 8px 0;
        }
        .metric-value { font-size: 2.5rem; font-weight: bold; color: #00d4ff; }
        .metric-label { font-size: 0.9rem; color: #aaa; margin-top: 4px; }
        .stSelectbox label { color: #7eb8ff !important; }
        </style>
    """, unsafe_allow_html=True)

    st.sidebar.markdown("## 🏙️ CityScape Segmentation")
    st.sidebar.markdown("---")
    page = st.sidebar.radio("Navigate", ["📊 Training Dashboard", "🖼️ Prediction Demo"])

    if page == "📊 Training Dashboard":
        show_training_page()
    else:
        show_prediction_page()


def show_training_page():
    st.title("📊 Training Dashboard")
    st.markdown("### CityScape UNet Segmentation — Training Metrics")

    metrics_path = os.path.join(SAVE_DIR, "metrics.json")
    if not os.path.exists(metrics_path):
        st.error("No metrics found. Please run `train.py` first.")
        return

    metrics = load_metrics()
    test_miou = metrics["test_miou"]
    test_mdice = metrics["test_mdice"]
    train_losses = metrics["train_losses"]
    train_mious = metrics["train_mious"]
    train_mdices = metrics["train_mdices"]

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-value'>{'✅' if test_miou >= 0.48 else '⚠️'} {test_miou:.4f}</div>
            <div class='metric-label'>Test mIoU</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-value'>{'✅' if test_mdice >= 0.48 else '⚠️'} {test_mdice:.4f}</div>
            <div class='metric-label'>Test mDice</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-value'>{len(train_losses)}</div>
            <div class='metric-label'>Epochs Trained</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Training Curves")

    epochs = list(range(1, len(train_losses) + 1))
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.patch.set_facecolor('#0e1117')

    for ax in axes:
        ax.set_facecolor('#1a1f2e')
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')
        for spine in ax.spines.values():
            spine.set_edgecolor('#334')

    axes[0].plot(epochs, train_losses, color='#00d4ff', linewidth=2, marker='o', markersize=4)
    axes[0].fill_between(epochs, train_losses, alpha=0.15, color='#00d4ff')
    axes[0].set_title("Training Loss", fontsize=13)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.2)

    axes[1].plot(epochs, train_mious, color='#00ff88', linewidth=2, marker='s', markersize=4)
    axes[1].fill_between(epochs, train_mious, alpha=0.15, color='#00ff88')
    axes[1].set_title("Training mIoU", fontsize=13)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("mIoU")
    axes[1].grid(True, alpha=0.2)

    axes[2].plot(epochs, train_mdices, color='#ff6b6b', linewidth=2, marker='^', markersize=4)
    axes[2].fill_between(epochs, train_mdices, alpha=0.15, color='#ff6b6b')
    axes[2].set_title("Training mDice", fontsize=13)
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("mDice")
    axes[2].grid(True, alpha=0.2)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("---")
    plots_path = os.path.join(SAVE_DIR, "training_plots.png")
    if os.path.exists(plots_path):
        st.markdown("### Saved Training Plots")
        st.image(plots_path, use_column_width=True)

    st.markdown(f"""
    <div style='background:#1e2a3a; border-radius:10px; padding:16px; margin-top:16px; border:1px solid #00d4ff33;'>
        <h4 style='color:#00d4ff'>📋 Final Test Set Results</h4>
        <p style='color:#ccc; font-size:1.1rem;'>
            <b>mIoU:</b> <span style='color:#00ff88'>{test_miou:.4f}</span> &nbsp;&nbsp;&nbsp;
            <b>mDice:</b> <span style='color:#ff6b6b'>{test_mdice:.4f}</span>
        </p>
        <p style='color:#aaa; font-size:0.85rem;'>
            {'Both metrics are above 0.48 ✅' if test_miou >= 0.48 and test_mdice >= 0.48 else '⚠️ One or more metrics below 0.48'}
        </p>
    </div>
    """, unsafe_allow_html=True)


def show_prediction_page():
    st.title("🖼️ Segmentation Prediction Demo")
    st.markdown("Upload up to **4 images** from the test set to see ground-truth vs predicted masks.")

    model_path = os.path.join(SAVE_DIR, "unet_cityscape.pth")
    if not os.path.exists(model_path):
        st.error("Model not found. Please run `train.py` first.")
        return

    model, device = load_model()

    test_imgs_path = os.path.join(SAVE_DIR, "test_imgs.npy")
    test_masks_path = os.path.join(SAVE_DIR, "test_masks.npy")

    if os.path.exists(test_imgs_path):
        test_imgs = np.load(test_imgs_path, allow_pickle=True).tolist()
        test_masks = np.load(test_masks_path, allow_pickle=True).tolist()
        st.info(f"Test set has **{len(test_imgs)}** images. You can upload any test image below.")

    uploaded_files = st.file_uploader(
        "Upload test images (RGB)", type=["png", "jpg", "jpeg"],
        accept_multiple_files=True
    )

    if uploaded_files:
        files_to_use = uploaded_files[:4]
        if len(uploaded_files) > 4:
            st.warning("Only the first 4 images will be processed.")

        st.markdown("---")
        st.markdown("### Results")

        for i, uploaded_file in enumerate(files_to_use):
            img_pil = Image.open(uploaded_file).convert("RGB")
            img_name = uploaded_file.name

            base_name = os.path.splitext(img_name)[0]
            gt_mask_pil = None
            if os.path.exists(test_imgs_path):
                for idx, tp in enumerate(test_imgs):
                    if base_name in tp:
                        try:
                            gt_mask_pil = Image.open(test_masks[idx])
                        except Exception:
                            pass
                        break

            pred_label = predict(model, device, img_pil)
            pred_color = label_to_color(pred_label)

            st.markdown(f"#### Image {i+1}: `{img_name}`")
            cols = st.columns(3 if gt_mask_pil else 2)

            with cols[0]:
                st.markdown("**Input Image**")
                st.image(img_pil, use_column_width=True)

            if gt_mask_pil:
                with cols[1]:
                    st.markdown("**Ground Truth Mask**")
                    gt_resized = gt_mask_pil.resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)
                    gt_label = encode_mask(gt_resized)
                    gt_color = label_to_color(gt_label)
                    st.image(gt_color, use_column_width=True)

                with cols[2]:
                    st.markdown("**Predicted Mask**")
                    st.image(pred_color, use_column_width=True)
            else:
                with cols[1]:
                    st.markdown("**Predicted Mask**")
                    st.image(pred_color, use_column_width=True)

            st.markdown("---")

        st.markdown("### 🎨 Class Color Legend")
        cols = st.columns(4)
        for i, (name, color) in enumerate(zip(CLASS_NAMES, COLORS)):
            with cols[i % 4]:
                hex_color = "#{:02x}{:02x}{:02x}".format(*color)
                st.markdown(
                    f"<div style='display:flex;align-items:center;gap:6px;margin:2px 0;'>"
                    f"<div style='width:16px;height:16px;background:{hex_color};border-radius:3px;border:1px solid #444'></div>"
                    f"<span style='color:#ccc;font-size:0.8rem;'>{name}</span></div>",
                    unsafe_allow_html=True
                )
    else:
        st.markdown("""
        <div style='text-align:center; padding:40px; background:#1a1f2e; border-radius:12px; border:2px dashed #00d4ff55;'>
            <h3 style='color:#00d4ff'>⬆️ Upload Test Images</h3>
            <p style='color:#888'>Drag and drop or click to upload up to 4 RGB images from the test set</p>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
