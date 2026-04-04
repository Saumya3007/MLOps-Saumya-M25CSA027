"""
Docker: python main.py
Flags:  --skip_grid   (skip LoRA 9-combo grid)
        --skip_optuna (skip Optuna search)
        --epochs N    (override epoch count)
"""
import os, sys, json, argparse, itertools
import wandb
import optuna
import torch
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import (
    WANDB_KEY, WANDB_PROJECT, WANDB_ENTITY,
    LORA_RANKS, LORA_ALPHAS, LORA_DROPOUT, NUM_CLASSES,
    WEIGHTS_DIR, RESULTS_DIR, PLOTS_DIR, DEVICE, NUM_EPOCHS, OPTUNA_TRIALS
)
from src.dataset import get_dataloaders, CIFAR100_CLASSES
from src.model   import build_vit_baseline, build_vit_lora, count_trainable_params
from src.trainer import run_experiment, evaluate_classwise
from src.plots   import (
    plot_loss_acc,
    save_epoch_table_png,
    plot_classwise_histogram,
    plot_all_val_acc_comparison,
    plot_lora_heatmap,
    save_test_summary_table,
)
from src.upload  import push_model_to_hub

parser = argparse.ArgumentParser()
parser.add_argument("--skip_grid",   action="store_true")
parser.add_argument("--skip_optuna", action="store_true")
parser.add_argument("--epochs",      type=int, default=NUM_EPOCHS)
args   = parser.parse_args()
EPOCHS = args.epochs

os.makedirs(WEIGHTS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR,   exist_ok=True)

if WANDB_KEY:
    wandb.login(key=WANDB_KEY)

print("=" * 65)
print("Loading CIFAR-100 …")
train_loader, val_loader, test_loader = get_dataloaders()
device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

all_results  = {}  
test_table   = []  
grid_results = {}  
print("\n" + "=" * 65)
print("Q1-A  Baseline (head-only, no LoRA)")
model_bl     = build_vit_baseline()
n_train_bl   = count_trainable_params(model_bl)
print(f"  Trainable params: {n_train_bl:,}")

history_bl, best_val_bl, _ = run_experiment(
    model_bl, train_loader, val_loader,
    experiment_name="baseline_no_lora",
    use_lora=False, epochs=EPOCHS
)
all_results["baseline_no_lora"] = best_val_bl

plot_loss_acc("baseline_no_lora")
save_epoch_table_png("baseline_no_lora")         
ckpt = torch.load(
    os.path.join(WEIGHTS_DIR, "baseline_no_lora_best.pt"), map_location=device)
model_bl.load_state_dict(ckpt["state_dict"])
model_bl = model_bl.to(device)
overall_bl, per_class_bl = evaluate_classwise(model_bl, test_loader, NUM_CLASSES, device)
plot_classwise_histogram("baseline_no_lora", per_class_bl)
test_table.append({
    "Experiment":           "baseline_no_lora",
    "LoRA":                 "without",
    "Rank":                 "-",
    "Alpha":                "-",
    "Dropout":              "-",
    "Overall Test Acc (%)": round(overall_bl * 100, 2),
    "Trainable Params":     n_train_bl,
})
print(f"  Baseline Test Accuracy: {overall_bl*100:.2f}%")

# ════════════════════════════════════════════════════════════════════════════
# Q1-B  LoRA GRID  rank ∈ {2,4,8}  ×  alpha ∈ {2,4,8}  dropout=0.1
# ════════════════════════════════════════════════════════════════════════════
if not args.skip_grid:
    combos = list(itertools.product(LORA_RANKS, LORA_ALPHAS))
    print(f"\nQ1-B  LoRA grid: {len(combos)} experiments")

    for exp_no, (rank, alpha) in enumerate(combos, start=1):
        exp_name = f"lora_r{rank}_a{alpha}"
        print(f"\n{'='*65}")
        print(f"  Experiment {exp_no}/{len(combos)}: {exp_name}  "
              f"(rank={rank}, alpha={alpha}, dropout={LORA_DROPOUT})")
        model_lora = build_vit_lora(rank=rank, alpha=alpha, dropout=LORA_DROPOUT)
        n_train    = count_trainable_params(model_lora)
        print(f"  Trainable params: {n_train:,}")

        history, best_val, _ = run_experiment(
            model_lora, train_loader, val_loader,
            experiment_name=exp_name,
            rank=rank, alpha=alpha, dropout=LORA_DROPOUT,
            use_lora=True, epochs=EPOCHS
        )
        all_results[exp_name]       = best_val
        grid_results[(rank, alpha)] = best_val

        plot_loss_acc(exp_name)
        save_epoch_table_png(exp_name, rank=rank, alpha=alpha)  

        ckpt = torch.load(
            os.path.join(WEIGHTS_DIR, f"{exp_name}_best.pt"), map_location=device)
        model_lora.load_state_dict(ckpt["state_dict"])
        model_lora = model_lora.to(device)
        overall, per_class = evaluate_classwise(
            model_lora, test_loader, NUM_CLASSES, device)
        plot_classwise_histogram(exp_name, per_class)
        test_table.append({
            "Experiment":           exp_name,
            "LoRA":                 "with",
            "Rank":                 rank,
            "Alpha":                alpha,
            "Dropout":              LORA_DROPOUT,
            "Overall Test Acc (%)": round(overall * 100, 2),
            "Trainable Params":     n_train,
        })
        print(f"  Test Accuracy: {overall*100:.2f}%")

    plot_lora_heatmap(grid_results)

# ════════════════════════════════════════════════════════════════════════════
# Q1-C  OPTUNA  hyperparameter search on LoRA params
# ════════════════════════════════════════════════════════════════════════════
if not args.skip_optuna:
    print("\n" + "=" * 65)
    print(f"Q1-C  Optuna search  ({OPTUNA_TRIALS} trials)")

    def optuna_objective(trial):
        r   = trial.suggest_categorical("rank",    [2, 4, 8, 16])
        a   = trial.suggest_categorical("alpha",   [4, 8, 16, 32])
        do  = trial.suggest_float("dropout",  0.0, 0.3, step=0.05)
        lr_ = trial.suggest_float("lr",       1e-4, 5e-3, log=True)
        m   = build_vit_lora(rank=r, alpha=a, dropout=do).to(device)
        crit = torch.nn.CrossEntropyLoss()
        opt_ = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, m.parameters()),
            lr=lr_, weight_decay=1e-4)
        from src.trainer import train_one_epoch, evaluate
        for _ in range(3):                        # 3-epoch proxy
            train_one_epoch(m, train_loader, crit, opt_, device)
        _, val_acc = evaluate(m, val_loader, crit, device)
        return val_acc

    study = optuna.create_study(direction="maximize", study_name="vit_lora_optuna")
    study.optimize(optuna_objective, n_trials=OPTUNA_TRIALS, show_progress_bar=True)
    bp = study.best_params
    print(f"\n  Best params: {bp}")
    print(f"  Best proxy val acc: {study.best_value*100:.2f}%")
    with open(os.path.join(RESULTS_DIR, "optuna_results.json"), "w") as f:
        json.dump({"best_params": bp, "best_val_acc": study.best_value}, f, indent=2)

    print("\n  Training Optuna-best model (full epochs) …")
    model_opt  = build_vit_lora(rank=bp["rank"], alpha=bp["alpha"], dropout=bp["dropout"])
    n_train_opt = count_trainable_params(model_opt)
    history_opt, best_val_opt, _ = run_experiment(
        model_opt, train_loader, val_loader,
        experiment_name="optuna_best",
        rank=bp["rank"], alpha=bp["alpha"], dropout=bp["dropout"],
        use_lora=True, epochs=EPOCHS
    )
    all_results["optuna_best"] = best_val_opt
    plot_loss_acc("optuna_best")
    save_epoch_table_png("optuna_best", rank=bp["rank"], alpha=bp["alpha"])

    ckpt = torch.load(
        os.path.join(WEIGHTS_DIR, "optuna_best_best.pt"), map_location=device)
    model_opt.load_state_dict(ckpt["state_dict"])
    model_opt = model_opt.to(device)
    overall_opt, per_class_opt = evaluate_classwise(
        model_opt, test_loader, NUM_CLASSES, device)
    plot_classwise_histogram("optuna_best", per_class_opt)
    test_table.append({
        "Experiment":           "optuna_best",
        "LoRA":                 "with",
        "Rank":                 bp["rank"],
        "Alpha":                bp["alpha"],
        "Dropout":              bp["dropout"],
        "Overall Test Acc (%)": round(overall_opt * 100, 2),
        "Trainable Params":     n_train_opt,
    })
    print(f"  Optuna Best Test Accuracy: {overall_opt*100:.2f}%")

    best_ckpt_path = os.path.join(WEIGHTS_DIR, "optuna_best_best.pt")
    hf_url = push_model_to_hub(best_ckpt_path, repo_name="vit-s-cifar100-lora-best")
    print(f"  HuggingFace: {hf_url}")

print("\n" + "=" * 65)
print("Saving final tables and comparison plots …")

csv_path, png_path = save_test_summary_table(test_table)

plot_all_val_acc_comparison(all_results)

col_headers = ["Experiment", "LoRA", "Rank", "Alpha",
               "Dropout", "Test Acc (%)", "Trainable Params"]
widths      = [24, 8, 6, 6, 8, 14, 20]
sep = "+" + "+".join("-" * w for w in widths) + "+"
fmt = "|" + "|".join(f"{{:<{w}}}" for w in widths) + "|"
print(sep)
print(fmt.format(*col_headers))
print(sep)
for row in test_table:
    print(fmt.format(
        row["Experiment"], row["LoRA"], str(row["Rank"]),
        str(row["Alpha"]), str(row["Dropout"]),
        str(row["Overall Test Acc (%)"]),
        f"{row['Trainable Params']:,}"
    ))
print(sep)

print(f"""
✅  Pipeline complete!
   Weights  → {WEIGHTS_DIR}/  (1 _best.pt per experiment)
   Results  → {RESULTS_DIR}/  (JSON histories + test_summary_table.csv)
   Plots    → {PLOTS_DIR}/
               ├── <exp>_curves.png          (loss & acc curves)
               ├── <exp>_epoch_table.png     (per-epoch train/val table)
               ├── <exp>_classwise.png       (class-wise histogram)
               ├── lora_heatmap.png          (rank × alpha grid)
               ├── all_experiments_val_acc.png
               └── test_summary_table.png    (assignment test table)
""")