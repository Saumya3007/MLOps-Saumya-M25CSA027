import os, json, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

try:
    from src.config import PLOTS_DIR, RESULTS_DIR
except ImportError:
    _BASE       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    PLOTS_DIR   = os.path.join(_BASE, "plots")
    RESULTS_DIR = os.path.join(_BASE, "results")

try:
    from src.dataset import CIFAR100_CLASSES
except ImportError:
    CIFAR100_CLASSES = [str(i) for i in range(100)]

os.makedirs(PLOTS_DIR,   exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_history(experiment_name: str):
    path = os.path.join(RESULTS_DIR, f"{experiment_name}_history.json")
    with open(path) as f:
        return json.load(f)["history"]


def plot_loss_acc(experiment_name: str):
    history  = load_history(experiment_name)
    epochs   = [h["epoch"]      for h in history]
    tr_loss  = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"]   for h in history]
    tr_acc   = [h["train_acc"]  for h in history]
    val_acc  = [h["val_acc"]    for h in history]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    axes[0].plot(epochs, tr_loss,  "o-",  label="Train Loss", color="#2980b9", linewidth=2)
    axes[0].plot(epochs, val_loss, "s--", label="Val Loss",   color="#e74c3c", linewidth=2)
    axes[0].set_title(f"{experiment_name} — Loss", fontweight="bold")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, tr_acc,  "o-",  label="Train Acc", color="#27ae60", linewidth=2)
    axes[1].plot(epochs, val_acc, "s--", label="Val Acc",   color="#e67e22", linewidth=2)
    axes[1].set_title(f"{experiment_name} — Accuracy", fontweight="bold")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, f"{experiment_name}_curves.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  [plot]  {out}")
    return out


def save_epoch_table_png(experiment_name: str, rank=None, alpha=None):
    history = load_history(experiment_name)
    cols = ["Epoch", "Training Loss", "Validation Loss",
            "Training Accuracy", "Validation Accuracy"]
    rows = [
        [h["epoch"],
         f"{h['train_loss']:.4f}",
         f"{h['val_loss']:.4f}",
         f"{h['train_acc']:.2f}%",
         f"{h['val_acc']:.2f}%"]
        for h in history
    ]
    n = len(rows)
    fig, ax = plt.subplots(figsize=(11, max(3.0, 0.42 * (n + 2))))
    ax.axis("off")
    title = f"Experiment: {experiment_name}"
    if rank is not None:
        title += f"   |   Rank: {rank}   Alpha: {alpha}"
    ax.set_title(title, fontsize=11, fontweight="bold", pad=12)

    tbl = ax.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(9.5)
    tbl.auto_set_column_width(col=list(range(len(cols))))
    for j in range(len(cols)):
        tbl[(0, j)].set_facecolor("#2c3e50")
        tbl[(0, j)].set_text_props(color="white", fontweight="bold")
    for i in range(1, n + 1):
        bg = "#eaf2fb" if i % 2 == 0 else "white"
        for j in range(len(cols)):
            tbl[(i, j)].set_facecolor(bg)

    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, f"{experiment_name}_epoch_table.png")
    plt.savefig(out, dpi=160, bbox_inches="tight"); plt.close()
    print(f"  [table] {out}")
    return out


def plot_classwise_histogram(experiment_name: str, per_class_acc: list):
    fig, ax = plt.subplots(figsize=(24, 5))
    colors  = ["#2ecc71" if a >= 0.6 else "#e67e22" if a >= 0.4 else "#e74c3c"
               for a in per_class_acc]
    ax.bar(range(len(CIFAR100_CLASSES)), [a * 100 for a in per_class_acc], color=colors)
    ax.set_xticks(range(len(CIFAR100_CLASSES)))
    ax.set_xticklabels(CIFAR100_CLASSES, rotation=90, fontsize=6)
    mean_acc = float(np.mean(per_class_acc)) * 100
    ax.axhline(mean_acc, color="blue", linestyle="--", linewidth=1.5,
               label=f"Mean {mean_acc:.1f}%")
    patches = [
        mpatches.Patch(color="#2ecc71", label="≥60%"),
        mpatches.Patch(color="#e67e22", label="40–60%"),
        mpatches.Patch(color="#e74c3c", label="<40%"),
        plt.Line2D([0],[0], color="blue", linestyle="--",
                   label=f"Mean {mean_acc:.1f}%"),
    ]
    ax.legend(handles=patches, fontsize=9)
    ax.set_title(f"{experiment_name} — Class-wise Test Accuracy", fontweight="bold")
    ax.set_xlabel("CIFAR-100 Class"); ax.set_ylabel("Accuracy (%)")
    ax.grid(axis="y", alpha=0.3); plt.tight_layout()
    out = os.path.join(PLOTS_DIR, f"{experiment_name}_classwise.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  [plot]  {out}")
    return out


# ── 4.  LoRA Gradient Norm Curves (from saved grad_history in JSON) ───────────
def plot_grad_norms(experiment_name: str):
    path = os.path.join(RESULTS_DIR, f"{experiment_name}_history.json")
    with open(path) as f:
        data = json.load(f)
    grad_hist = data.get("grad_history", {})
    out = os.path.join(PLOTS_DIR, f"{experiment_name}_grad_norms.png")
    if not grad_hist:
        print(f"  [skip]  No grad_history in JSON for {experiment_name}")
        return None
    fig, ax = plt.subplots(figsize=(13, 4))
    for layer, vals in grad_hist.items():
        short = layer.split("base_model.")[-1]
        ax.plot(vals, label=short, alpha=0.85, linewidth=1.2)
    ax.set_title(f"{experiment_name} — LoRA Weight Gradient Norms per Epoch",
                 fontweight="bold")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Gradient Norm")
    ax.legend(fontsize=5, ncol=3); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  [plot]  {out}")
    return out


# ── 5.  All-experiments Val Accuracy Bar Chart ────────────────────────────────
def plot_all_val_acc_comparison(results: dict):
    names = list(results.keys())
    accs  = [results[n] * 100 if results[n] <= 1 else results[n] for n in names]
    fig, ax = plt.subplots(figsize=(max(10, len(names) * 0.9), 5))
    cmap = plt.cm.viridis(np.linspace(0.2, 0.85, len(names)))
    bars = ax.bar(names, accs, color=cmap)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.2,
                f"{acc:.1f}%", ha="center", va="bottom",
                fontsize=7.5, fontweight="bold")
    ax.set_title("Best Validation Accuracy — All Experiments",
                 fontsize=12, fontweight="bold")
    ax.set_ylabel("Val Accuracy (%)"); ax.set_ylim(0, max(accs) + 6)
    ax.set_xticklabels(names, rotation=40, ha="right", fontsize=8)
    ax.grid(axis="y", alpha=0.3); plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "all_experiments_val_acc.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  [plot]  {out}")
    return out


# ── 6.  LoRA Rank × Alpha Heatmap ─────────────────────────────────────────────
def plot_lora_heatmap(grid_results: dict):
    ranks  = sorted(set(k[0] for k in grid_results))
    alphas = sorted(set(k[1] for k in grid_results))
    matrix = np.array([
        [grid_results.get((r, a), 0) * 100
         if grid_results.get((r, a), 0) <= 1
         else grid_results.get((r, a), 0)
         for a in alphas]
        for r in ranks
    ])
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(matrix,
                xticklabels=[f"α={a}" for a in alphas],
                yticklabels=[f"r={r}" for r in ranks],
                annot=True, fmt=".1f", cmap="YlOrRd", ax=ax,
                cbar_kws={"label": "Val Acc (%)"})
    ax.set_title("LoRA Grid — Best Val Accuracy (%)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Alpha"); ax.set_ylabel("Rank")
    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "lora_heatmap.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  [plot]  {out}")
    return out


# ── 7.  Test Summary Table — PNG + CSV ────────────────────────────────────────
def save_test_summary_table(test_table: list):
    cols = ["Experiment", "LoRA Layers (with/without)", "Rank", "Alpha",
            "Dropout", "Overall Test Accuracy", "Trainable Parameters used"]

    # CSV
    csv_path = os.path.join(RESULTS_DIR, "test_summary_table.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for row in test_table:
            writer.writerow({
                "Experiment":                 row["Experiment"],
                "LoRA Layers (with/without)": row["LoRA"],
                "Rank":                       row["Rank"],
                "Alpha":                      row["Alpha"],
                "Dropout":                    row["Dropout"],
                "Overall Test Accuracy":      f"{row['Overall Test Acc (%)']:.2f}%",
                "Trainable Parameters used":  f"{row['Trainable Params']:,}",
            })
    print(f"  [csv]   {csv_path}")

    # PNG
    rows_data = [
        [row["Experiment"], row["LoRA"], str(row["Rank"]), str(row["Alpha"]),
         str(row["Dropout"]), f"{row['Overall Test Acc (%)']:.2f}%",
         f"{row['Trainable Params']:,}"]
        for row in test_table
    ]
    n = len(rows_data)
    fig, ax = plt.subplots(figsize=(17, max(2.5, 0.48 * (n + 2))))
    ax.axis("off")
    ax.set_title("Test Results Summary Table", fontsize=13, fontweight="bold", pad=14)
    tbl = ax.table(cellText=rows_data, colLabels=cols, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(8.5)
    tbl.auto_set_column_width(col=list(range(len(cols))))
    for j in range(len(cols)):
        tbl[(0, j)].set_facecolor("#1a252f")
        tbl[(0, j)].set_text_props(color="white", fontweight="bold")
    for i, rv in enumerate(rows_data, start=1):
        if "optuna" in rv[0]:    bg = "#d5e8d4"
        elif rv[1] == "with":    bg = "#dae8fc"
        else:                    bg = "#fff2cc"
        alt = "#f9f9f9" if i % 2 == 0 else bg
        for j in range(len(cols)):
            tbl[(i, j)].set_facecolor(alt)
    plt.tight_layout()
    png_path = os.path.join(PLOTS_DIR, "test_summary_table.png")
    plt.savefig(png_path, dpi=160, bbox_inches="tight"); plt.close()
    print(f"  [table] {png_path}")
    return csv_path, png_path