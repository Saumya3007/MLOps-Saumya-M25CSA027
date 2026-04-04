"""
upload_to_wandb.py
------------------
Skips ALL training. Reads saved _history.json + existing PNGs,
generates any missing plots, then uploads EVERYTHING to WandB.

Usage:
    python upload_to_wandb.py
"""
import os, sys, json, glob, csv
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import wandb

from src.config import (WANDB_KEY, WANDB_PROJECT, WANDB_ENTITY,
                        RESULTS_DIR, PLOTS_DIR)
from src.plots  import (
    plot_loss_acc,
    save_epoch_table_png,
    plot_grad_norms,
    plot_all_val_acc_comparison,
    plot_lora_heatmap,
)

# ── WandB login ───────────────────────────────────────────────────────────────
if WANDB_KEY:
    wandb.login(key=WANDB_KEY)
else:
    print("[WARN] WANDB_KEY not set – runs may fail if not already logged in")

os.makedirs(PLOTS_DIR,   exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def _png(name):
    p = os.path.join(PLOTS_DIR, name)
    return p if os.path.exists(p) else None


# ── Discover all completed experiments ───────────────────────────────────────
json_files = sorted(glob.glob(os.path.join(RESULTS_DIR, "*_history.json")))
if not json_files:
    print(f"[ERROR] No *_history.json found in {RESULTS_DIR}")
    sys.exit(1)

print(f"\nFound {len(json_files)} experiments:")
for f in json_files:
    print(f"  {os.path.basename(f)}")

all_best_val = {}   # exp_name → best_val_acc (0-1 scale)
grid_results = {}   # (rank, alpha) → best_val_acc


# ════════════════════════════════════════════════════════════════════════════
# PER-EXPERIMENT: generate plots + upload
# ════════════════════════════════════════════════════════════════════════════
for jf in json_files:
    with open(jf) as f:
        data = json.load(f)

    exp_name = data["experiment"]
    history  = data["history"]
    use_lora = (exp_name != "baseline_no_lora")

    # ── Parse rank / alpha ───────────────────────────────────────────────────
    rank, alpha, dropout = None, None, None
    if "lora_r" in exp_name:
        try:
            parts = exp_name.replace("lora_r", "").split("_a")
            rank, alpha = int(parts[0]), int(parts[1])
            dropout = 0.1
        except Exception:
            pass
    elif exp_name == "optuna_best":
        opt_path = os.path.join(RESULTS_DIR, "optuna_results.json")
        if os.path.exists(opt_path):
            bp      = json.load(open(opt_path))["best_params"]
            rank    = bp.get("rank")
            alpha   = bp.get("alpha")
            dropout = bp.get("dropout")

    # ── best_val_acc – normalise to 0-1 ─────────────────────────────────────
    bv = data.get("best_val_acc", max(h["val_acc"] for h in history))
    best_val = bv / 100 if bv > 1 else bv
    all_best_val[exp_name] = best_val
    if rank and alpha:
        grid_results[(rank, alpha)] = best_val

    print(f"\n{'='*60}")
    print(f"  Processing: {exp_name}  (best_val={best_val*100:.2f}%)")

    # ── Generate / refresh all plots ─────────────────────────────────────────
    curves_path    = plot_loss_acc(exp_name)
    table_path     = save_epoch_table_png(exp_name, rank=rank, alpha=alpha)
    grad_path      = plot_grad_norms(exp_name) if use_lora else None   # None if no grad_history
    classwise_path = _png(f"{exp_name}_classwise.png")                 # already saved by main.py

    # ── Open WandB run ───────────────────────────────────────────────────────
    run = wandb.init(
        project=WANDB_PROJECT,
        **({"entity": WANDB_ENTITY} if WANDB_ENTITY else {}),
        name=exp_name,
        config=dict(experiment=exp_name, rank=rank, alpha=alpha,
                    dropout=dropout, use_lora=use_lora,
                    best_val_acc=round(best_val * 100, 2)),
        reinit="finish_previous",
        tags=["upload", "lora" if use_lora else "baseline"]
    )

    # ── Log epoch scalars → creates WandB line charts ───────────────────────
    wb_table = wandb.Table(
        columns=["Epoch", "Train Loss", "Val Loss", "Train Acc (%)", "Val Acc (%)"])
    for h in history:
        run.log({
            "train/loss": h["train_loss"],
            "train/acc":  h["train_acc"] / 100,
            "val/loss":   h["val_loss"],
            "val/acc":    h["val_acc"]   / 100,
            "epoch":      h["epoch"],
        })
        wb_table.add_data(
            h["epoch"],
            h["train_loss"], h["val_loss"],
            h["train_acc"],  h["val_acc"]
        )

    # ── Log all images + table ───────────────────────────────────────────────
    img_log = {
        "plots/loss_acc_curves":    wandb.Image(curves_path,
                                     caption=f"{exp_name} Loss & Accuracy"),
        "plots/epoch_table":        wandb.Image(table_path,
                                     caption=f"{exp_name} Per-Epoch Table"),
        "tables/train_val_metrics": wb_table,
    }
    if classwise_path:
        img_log["plots/classwise_histogram"] = wandb.Image(
            classwise_path, caption=f"{exp_name} Class-wise Test Accuracy")
    else:
        print(f"  [warn]  No classwise PNG found for {exp_name} – skipping")

    if grad_path:
        img_log["plots/lora_grad_norms"] = wandb.Image(
            grad_path, caption=f"{exp_name} LoRA Gradient Norms")

    run.log(img_log)
    run.finish()
    print(f"  ✓ Uploaded: {exp_name}")


# ════════════════════════════════════════════════════════════════════════════
# GLOBAL SUMMARY: heatmap + comparison bar + test table → single WandB run
# ════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("  Generating & uploading global summary ...")

comparison_path = plot_all_val_acc_comparison(all_best_val)
heatmap_path    = plot_lora_heatmap(grid_results) if grid_results else None

run = wandb.init(
    project=WANDB_PROJECT,
    **({"entity": WANDB_ENTITY} if WANDB_ENTITY else {}),
    name="summary_all_experiments",
    reinit="finish_previous",
    tags=["summary"]
)

summary_log = {
    "summary/val_acc_comparison": wandb.Image(comparison_path,
                                   caption="All Experiments – Best Val Acc"),
}
if heatmap_path:
    summary_log["summary/lora_heatmap"] = wandb.Image(heatmap_path,
                                            caption="LoRA Rank × Alpha Heatmap")

test_summary_png = _png("test_summary_table.png")
if test_summary_png:
    summary_log["summary/test_table_image"] = wandb.Image(test_summary_png,
                                               caption="Test Results Summary Table")

# Upload CSV as interactive WandB Table
test_csv = os.path.join(RESULTS_DIR, "test_summary_table.csv")
if os.path.exists(test_csv):
    with open(test_csv) as f:
        reader = list(csv.DictReader(f))
    if reader:
        wb_t = wandb.Table(columns=list(reader[0].keys()))
        for row in reader:
            wb_t.add_data(*list(row.values()))
        summary_log["summary/test_results_table"] = wb_t
        print("  [wandb] Test CSV → WandB Table attached")
else:
    print(f"  [warn]  {test_csv} not found – run main.py first to generate it")

run.log(summary_log)
run.finish()

# ── Final summary printout ────────────────────────────────────────────────────
print(f"""
{'='*60}
✅  All plots generated & uploaded to WandB!

WandB project : https://wandb.ai/{WANDB_ENTITY or '<entity>'}/{WANDB_PROJECT}

Local plots saved in: {PLOTS_DIR}/
  <exp>_curves.png          ← Loss & Accuracy curves
  <exp>_epoch_table.png     ← Per-epoch train/val table
  <exp>_classwise.png       ← Class-wise test accuracy histogram
  <exp>_grad_norms.png      ← LoRA gradient norm curves (if available)
  lora_heatmap.png          ← Rank × Alpha grid heatmap
  all_experiments_val_acc.png
  test_summary_table.png
{'='*60}
""")