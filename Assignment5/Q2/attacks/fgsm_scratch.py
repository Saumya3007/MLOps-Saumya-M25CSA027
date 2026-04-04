import torch
import torch.nn as nn
import numpy as np


def fgsm_attack_scratch(model, images, labels, epsilon, device='cuda'):
    MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1,3,1,1).to(device)
    STD  = torch.tensor([0.2023, 0.1994, 0.2010]).view(1,3,1,1).to(device)

    images = images.clone().detach().to(device)
    labels = labels.clone().detach().to(device)

    images_norm = (images - MEAN) / STD
    images_norm.requires_grad_(True)

    model.eval()
    loss = nn.CrossEntropyLoss()(model(images_norm), labels)
    model.zero_grad()
    loss.backward()

    perturbed = images_norm + epsilon * images_norm.grad.data.sign()
    perturbed = perturbed * STD + MEAN
    perturbed = torch.clamp(perturbed, 0.0, 1.0)
    return perturbed.detach()


def evaluate_on_adversarial_scratch(model, images_np, labels_np, epsilon,
                                     device='cuda', batch_size=256):
    MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1,3,1,1).to(device)
    STD  = torch.tensor([0.2023, 0.1994, 0.2010]).view(1,3,1,1).to(device)

    all_adv, correct, total = [], 0, 0
    for i in range(0, len(images_np), batch_size):
        imgs = torch.tensor(images_np[i:i+batch_size]).float().to(device)
        lbls = torch.tensor(labels_np[i:i+batch_size]).long().to(device)
        adv  = fgsm_attack_scratch(model, imgs, lbls, epsilon, device)
        all_adv.append(adv.cpu().numpy())
        adv_norm = (adv - MEAN) / STD
        with torch.no_grad():
            outputs = model(adv_norm)
        correct += outputs.max(1)[1].eq(lbls).sum().item()
        total   += lbls.size(0)

    return 100.0 * correct / total, np.concatenate(all_adv, axis=0)