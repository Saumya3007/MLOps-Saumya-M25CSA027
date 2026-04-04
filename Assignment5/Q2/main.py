"""
Assignment 5 - Adversarial Attacks & Detection Pipeline
Run: python main.py [--skip_train] [--skip_attacks]
"""
import os, sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import argparse
import numpy as np
import torch
import wandb
from dotenv import load_dotenv

load_dotenv(os.path.join(_ROOT, ".env"))
WANDB_KEY    = os.environ.get("WANDB_API_KEY", "")
WANDB_PROJ   = os.environ.get("WANDB_PROJECT", "assignment5-adversarial")
WANDB_ENTITY = os.environ.get("WANDB_ENTITY",  None)
HF_TOKEN     = os.environ.get("HF_TOKEN",      "")
HF_REPO      = os.environ.get("HF_REPO_ID",    "")

CKPT_DIR      = os.path.join(_ROOT, "checkpoints")
PLOT_DIR      = os.path.join(_ROOT, "plots")
DATA_DIR      = os.path.join(_ROOT, "data")
RESNET18_CKPT = os.path.join(CKPT_DIR, "best_resnet18.pth")
ADV_CACHE     = os.path.join(CKPT_DIR, "adv_cache.npz")

for d in [CKPT_DIR, PLOT_DIR, DATA_DIR]:
    os.makedirs(d, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

from models.resnet_models     import get_resnet18
from utils.data_loader        import get_cifar10_loaders, get_raw_test_tensor
from utils.trainer            import train_classifier, evaluate
from attacks.fgsm_scratch     import evaluate_on_adversarial_scratch
from attacks.fgsm_art         import (build_art_classifier, fgsm_art_attack,
                                       pgd_art_attack, bim_art_attack, accuracy_on_adv)
from utils.visualize          import (plot_comparison, plot_epsilon_vs_accuracy,
                                       plot_accuracy_bar, log_wandb_samples)
from detectors.train_detector import build_and_train_detector


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--skip_train",   action="store_true")
    p.add_argument("--skip_attacks", action="store_true", help="Reuse cached adv arrays")
    p.add_argument("--epochs",       type=int,   default=100)
    p.add_argument("--det_epochs",   type=int,   default=60)
    p.add_argument("--epsilon",      type=float, default=4/255)
    p.add_argument("--n_samples",    type=int,   default=10000)
    return p.parse_args()


def _arr_acc(model, adv_np, labels_np, device, batch_size=256):
    MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1,3,1,1).to(device)
    STD  = torch.tensor([0.2023, 0.1994, 0.2010]).view(1,3,1,1).to(device)
    correct, total = 0, 0
    model.eval()
    for i in range(0, len(adv_np), batch_size):
        imgs = torch.tensor(adv_np[i:i+batch_size]).float().to(device)
        lbls = torch.tensor(labels_np[i:i+batch_size]).long().to(device)
        with torch.no_grad():
            out = model((imgs - MEAN) / STD)
        correct += out.max(1)[1].eq(lbls).sum().item()
        total   += lbls.size(0)
    return 100.0 * correct / total


def main():
    args = parse_args()
    wandb.login(key=WANDB_KEY)
    wandb.init(project=WANDB_PROJ,
               entity=WANDB_ENTITY if WANDB_ENTITY else None,
               name="assignment5-full-pipeline",
               config=vars(args))

    # ── STEP 1: Train ResNet-18 ─────────────────────────────────────────────
    print("\n" + "="*60)
    print("STEP 1: ResNet-18 on CIFAR-10  (target ≥ 72%)")
    print("="*60)

    train_loader, test_loader = get_cifar10_loaders(batch_size=128, data_dir=DATA_DIR)
    model18 = get_resnet18(num_classes=10)

    if args.skip_train and os.path.exists(RESNET18_CKPT):
        model18.load_state_dict(torch.load(RESNET18_CKPT,
                                            map_location=DEVICE, weights_only=True))
        clean_acc = evaluate(model18.to(DEVICE), test_loader, DEVICE)
        print(f"Loaded checkpoint — Clean acc: {clean_acc:.2f}%")
    else:
        clean_acc = train_classifier(model18, train_loader, test_loader,
                                     epochs=args.epochs, lr=0.1, device=DEVICE,
                                     save_path=RESNET18_CKPT)

    assert clean_acc >= 72.0, f"Clean acc {clean_acc:.2f}% < 72%!"
    wandb.log({"clean_accuracy": clean_acc})
    model18.load_state_dict(torch.load(RESNET18_CKPT,
                                        map_location=DEVICE, weights_only=True))
    model18 = model18.to(DEVICE).eval()

    # ── STEP 2: Adversarial examples ───────────────────────────────────────
    print("\n" + "="*60)
    print(f"STEP 2: Adversarial Examples  eps={args.epsilon*255:.2f}/255")
    print("="*60)

    raw_images, raw_labels = get_raw_test_tensor(data_dir=DATA_DIR)
    N = min(args.n_samples, len(raw_images))
    raw_images, raw_labels = raw_images[:N], raw_labels[:N]
    print(f"Using {N} test samples.")

    art_clf = build_art_classifier(model18, device=DEVICE)

    if args.skip_attacks and os.path.exists(ADV_CACHE):
        print("Loading cached adversarial arrays...")
        cache        = np.load(ADV_CACHE)
        adv_scratch  = cache['adv_scratch']
        adv_fgsm_art = cache['adv_fgsm_art']
        adv_pgd      = cache['adv_pgd']
        adv_bim      = cache['adv_bim']
        acc_scratch  = _arr_acc(model18, adv_scratch,  raw_labels, DEVICE)
        acc_fgsm_art = accuracy_on_adv(art_clf, adv_fgsm_art, raw_labels)
        acc_pgd      = accuracy_on_adv(art_clf, adv_pgd,      raw_labels)
        acc_bim      = accuracy_on_adv(art_clf, adv_bim,      raw_labels)
    else:
        print(f"\n[FGSM Scratch]  eps={args.epsilon*255:.2f}/255")
        acc_scratch, adv_scratch = evaluate_on_adversarial_scratch(
            model18, raw_images, raw_labels, epsilon=args.epsilon, device=DEVICE)

        print(f"\n[FGSM ART]  eps={args.epsilon*255:.2f}/255")
        adv_fgsm_art = fgsm_art_attack(art_clf, raw_images, epsilon=args.epsilon)
        acc_fgsm_art = accuracy_on_adv(art_clf, adv_fgsm_art, raw_labels)

        print(f"\n[PGD ART]  eps={args.epsilon*255:.2f}/255")
        adv_pgd = pgd_art_attack(art_clf, raw_images,
                                  epsilon=args.epsilon,
                                  eps_step=args.epsilon/4, max_iter=7)
        acc_pgd = accuracy_on_adv(art_clf, adv_pgd, raw_labels)

        print(f"\n[BIM ART]  eps={args.epsilon*255:.2f}/255")
        adv_bim = bim_art_attack(art_clf, raw_images,
                                  epsilon=args.epsilon,
                                  eps_step=args.epsilon/4, max_iter=7)
        acc_bim = accuracy_on_adv(art_clf, adv_bim, raw_labels)

        np.savez(ADV_CACHE,
                 adv_scratch=adv_scratch, adv_fgsm_art=adv_fgsm_art,
                 adv_pgd=adv_pgd, adv_bim=adv_bim)
        print("Adversarial arrays cached.")

    for name, acc in [("FGSM Scratch", acc_scratch), ("FGSM ART", acc_fgsm_art),
                       ("PGD ART",      acc_pgd),      ("BIM ART", acc_bim)]:
        print(f"  {name:<20} Accuracy: {acc:.2f}%")

    wandb.log({"fgsm_scratch_accuracy": acc_scratch, "fgsm_art_accuracy": acc_fgsm_art,
               "pgd_art_accuracy": acc_pgd, "bim_art_accuracy": acc_bim})

    # Epsilon sweep
    print("\n[Epsilon Sweep]")
    epsilons = [2/255, 4/255, 8/255, 12/255, 16/255]
    acc_s_list, acc_a_list = [], []
    for eps in epsilons:
        a_s, _ = evaluate_on_adversarial_scratch(
            model18, raw_images, raw_labels, epsilon=eps, device=DEVICE)
        a_a = accuracy_on_adv(art_clf,
                               fgsm_art_attack(art_clf, raw_images, epsilon=eps),
                               raw_labels)
        acc_s_list.append(a_s)
        acc_a_list.append(a_a)
        print(f"  eps={eps*255:.0f}/255  Scratch: {a_s:.2f}%  ART: {a_a:.2f}%")

    # ── STEP 3: Plots ───────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("STEP 3: Plots & WandB logging")
    print("="*60)

    plot_comparison(
        raw_images[:10], adv_scratch[:10], adv_fgsm_art[:10], raw_labels[:10],
        n=10, save_path=os.path.join(PLOT_DIR, "fgsm_comparison.png"))
    plot_epsilon_vs_accuracy(
        epsilons, acc_s_list, acc_a_list,
        save_path=os.path.join(PLOT_DIR, "epsilon_vs_accuracy.png"))
    plot_accuracy_bar(
        {"Clean": clean_acc, "FGSM Scratch": acc_scratch,
         "FGSM ART": acc_fgsm_art, "PGD ART": acc_pgd, "BIM ART": acc_bim},
        save_path=os.path.join(PLOT_DIR, "accuracy_comparison.png"))

    wandb.log({
        "plots/fgsm_comparison":
            wandb.Image(os.path.join(PLOT_DIR, "fgsm_comparison.png")),
        "plots/epsilon_vs_accuracy":
            wandb.Image(os.path.join(PLOT_DIR, "epsilon_vs_accuracy.png")),
        "plots/accuracy_comparison":
            wandb.Image(os.path.join(PLOT_DIR, "accuracy_comparison.png")),
    })
    log_wandb_samples(raw_images[:10], adv_scratch[:10], adv_fgsm_art[:10],
                      adv_pgd[:10], adv_bim[:10], raw_labels[:10])

    # ── STEP 4: Detectors ───────────────────────────────────────────────────
    print("\n" + "="*60)
    print("STEP 4: Adversarial Detectors  (target ≥ 70%)")
    print("="*60)

    print(f"\nUsing {N} clean + {N} adversarial samples for detection.")

    print("\n[Detector A] PGD Attack")
    _, pgd_acc = build_and_train_detector(
        "PGD", raw_images, adv_pgd,
        victim_model=model18, device=DEVICE,
        save_dir=CKPT_DIR, epochs=args.det_epochs)

    print("\n[Detector B] BIM Attack")
    _, bim_acc = build_and_train_detector(
        "BIM", raw_images, adv_bim,
        victim_model=model18, device=DEVICE,
        save_dir=CKPT_DIR, epochs=args.det_epochs)

    wandb.log({"detector_PGD_acc": pgd_acc, "detector_BIM_acc": bim_acc})

    # ── Final Summary ───────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("FINAL RESULTS SUMMARY")
    print("="*60)
    for name, val, threshold in [
        ("Clean Accuracy",        clean_acc,    72.0),
        ("FGSM Scratch Accuracy", acc_scratch,  None),
        ("FGSM ART Accuracy",     acc_fgsm_art, None),
        ("PGD ART Accuracy",      acc_pgd,      None),
        ("BIM ART Accuracy",      acc_bim,      None),
        ("Detector PGD Val Acc",  pgd_acc,      70.0),
        ("Detector BIM Val Acc",  bim_acc,      70.0),
    ]:
        if threshold:
            flag = "✓" if val >= threshold else f"✗ (need ≥{threshold}%)"
        else:
            flag = f"(drop={clean_acc - val:.1f}%)"
        print(f"  {name:<35} {val:>7.2f}%  {flag}")

    # ── HuggingFace upload ───────────────────────────────────────────────────
    if HF_TOKEN and HF_REPO:
        try:
            from huggingface_hub import HfApi, create_repo
            api = HfApi()
            # create_repo with exist_ok=True handles the 404 error
            create_repo(repo_id=HF_REPO, repo_type="model",
                        token=HF_TOKEN, exist_ok=True)
            for fname in os.listdir(CKPT_DIR):
                if fname.endswith(".pth"):
                    api.upload_file(
                        path_or_fileobj=os.path.join(CKPT_DIR, fname),
                        path_in_repo=fname,
                        repo_id=HF_REPO,
                        repo_type="model",
                        token=HF_TOKEN)
            print(f"\nModels uploaded to https://huggingface.co/{HF_REPO}")
        except Exception as e:
            print(f"HuggingFace upload failed: {e}")

    wandb.finish()
    print("\nPipeline complete!")


if __name__ == "__main__":
    main()