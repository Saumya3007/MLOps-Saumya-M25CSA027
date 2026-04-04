import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)


def get_cifar10_loaders(batch_size=128, num_workers=4, data_dir='./data'):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    train_ds = datasets.CIFAR10(root=data_dir, train=True,  download=True, transform=train_transform)
    test_ds  = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader


def get_raw_test_tensor(data_dir='./data'):
    """Return (images [0,1], labels) unnormalized — ART applies preprocessing internally."""
    transform = transforms.Compose([transforms.ToTensor()])
    test_ds = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)
    loader  = DataLoader(test_ds, batch_size=len(test_ds), shuffle=False, num_workers=4)
    images, labels = next(iter(loader))
    return images.numpy(), labels.numpy()


def make_detection_loader(clean_images, adv_images, batch_size=64):
    """Combine clean (label=0) and adversarial (label=1) into a binary detection dataset."""
    import numpy as np
    X = np.concatenate([clean_images, adv_images], axis=0)
    y = np.array([0]*len(clean_images) + [1]*len(adv_images), dtype=np.int64)
    ds = TensorDataset(torch.tensor(X).float(), torch.tensor(y).long())
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)