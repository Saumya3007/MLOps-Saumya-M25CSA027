import numpy as np
import torch.nn as nn
import torch
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import (FastGradientMethod,
                                  ProjectedGradientDescentPyTorch,
                                  BasicIterativeMethod)

CIFAR10_MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
CIFAR10_STD  = np.array([0.2023, 0.1994, 0.2010], dtype=np.float32)


def build_art_classifier(model, device='cuda'):
    classifier = PyTorchClassifier(
        model=model,
        loss=nn.CrossEntropyLoss(),
        optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
        input_shape=(3, 32, 32),
        nb_classes=10,
        preprocessing=(CIFAR10_MEAN.reshape(1,3,1,1),
                       CIFAR10_STD.reshape(1,3,1,1)),
        clip_values=(0.0, 1.0),
        device_type='gpu' if device == 'cuda' and torch.cuda.is_available() else 'cpu',
    )
    return classifier


def fgsm_art_attack(classifier, images_np, epsilon=4/255):
    attack = FastGradientMethod(estimator=classifier, eps=epsilon, batch_size=512)
    return attack.generate(x=images_np)


def pgd_art_attack(classifier, images_np, epsilon=4/255, eps_step=1/255, max_iter=7):
    attack = ProjectedGradientDescentPyTorch(
        estimator=classifier, eps=epsilon, eps_step=eps_step,
        max_iter=max_iter, targeted=False, batch_size=256
    )
    return attack.generate(x=images_np)


def bim_art_attack(classifier, images_np, epsilon=4/255, eps_step=1/255, max_iter=7):
    attack = BasicIterativeMethod(
        estimator=classifier, eps=epsilon, eps_step=eps_step,
        max_iter=max_iter, batch_size=256
    )
    return attack.generate(x=images_np)


def accuracy_on_adv(classifier, adv_images, labels):
    """Chunked evaluation — prevents ART predict() from hanging on large arrays."""
    chunk   = 500
    correct = 0
    total   = len(labels)
    for i in range(0, total, chunk):
        preds = np.argmax(
            classifier.predict(adv_images[i:i+chunk], batch_size=256), axis=1)
        correct += np.sum(preds == labels[i:i+chunk])
    return 100.0 * correct / total