import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import wandb
import os

CIFAR10_CLASSES = ['airplane','automobile','bird','cat','deer',
                   'dog','frog','horse','ship','truck']


def plot_comparison(clean_raw, adv_scratch, adv_art, labels, n=10,
                    save_path='plots/fgsm_comparison.png'):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, axes = plt.subplots(3, n, figsize=(n*2, 7))
    row_labels = ['Original', 'FGSM Scratch', 'FGSM ART']
    for col in range(n):
        for row, imgs in enumerate([clean_raw, adv_scratch, adv_art]):
            hwc = np.clip(imgs[col].transpose(1,2,0), 0, 1)
            axes[row, col].imshow(hwc)
            axes[row, col].axis('off')
            if row == 0:
                axes[row, col].set_title(CIFAR10_CLASSES[labels[col]], fontsize=8)
        axes[0, col].set_ylabel(row_labels[0] if col == 0 else '', fontsize=8)
    for row, lbl in enumerate(row_labels):
        axes[row, 0].set_ylabel(lbl, fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved {save_path}")


def plot_epsilon_vs_accuracy(epsilons, acc_scratch, acc_art,
                              save_path='plots/epsilon_vs_accuracy.png'):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(7, 5))
    plt.plot([e*255 for e in epsilons], acc_scratch, 'o-', label='FGSM Scratch', color='steelblue')
    plt.plot([e*255 for e in epsilons], acc_art,     's-', label='FGSM ART',     color='coral')
    plt.xlabel('Epsilon (out of 255)')
    plt.ylabel('Accuracy (%)')
    plt.title('Perturbation Strength vs Accuracy Drop')
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(save_path, dpi=150); plt.close()
    print(f"Saved {save_path}")


def plot_accuracy_bar(results_dict, save_path='plots/accuracy_comparison.png'):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    labels = list(results_dict.keys())
    values = list(results_dict.values())
    colors = ['#4CAF50','#2196F3','#FF5722','#9C27B0','#FF9800']
    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, values, color=colors[:len(labels)], edgecolor='white', linewidth=1.2)
    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f'{val:.1f}%', ha='center', va='bottom', fontsize=10)
    plt.ylabel('Accuracy (%)'); plt.title('Clean vs Adversarial Accuracy')
    plt.ylim(0, 110); plt.xticks(rotation=15, ha='right'); plt.tight_layout()
    plt.savefig(save_path, dpi=150); plt.close()
    print(f"Saved {save_path}")


def log_wandb_samples(clean_raw, adv_fgsm_scratch, adv_fgsm_art, adv_pgd, adv_bim, labels, n=10):
    def make_table(name, adv):
        table = wandb.Table(columns=['index','label','clean_image','adv_image'])
        for i in range(n):
            c = np.clip(clean_raw[i].transpose(1,2,0), 0, 1)
            a = np.clip(adv[i].transpose(1,2,0), 0, 1)
            table.add_data(i, CIFAR10_CLASSES[labels[i]],
                           wandb.Image(c, caption=f"Clean"),
                           wandb.Image(a, caption=f"Adv ({name})"))
        return table

    wandb.log({
        'samples/FGSM_Scratch': make_table('FGSM_Scratch', adv_fgsm_scratch),
        'samples/FGSM_ART':     make_table('FGSM_ART',     adv_fgsm_art),
        'samples/PGD':          make_table('PGD',           adv_pgd),
        'samples/BIM':          make_table('BIM',           adv_bim),
    })
    print("Logged WandB image sample tables.")