import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from src.config import IMG_SIZE, BATCH_SIZE, NUM_WORKERS

CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD  = (0.2675, 0.2565, 0.2761)

def get_transforms(train: bool):
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ])
    return transforms.Compose([
        transforms.Resize(IMG_SIZE + 16),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])

def get_dataloaders(data_root: str = "./data"):
    train_full = datasets.CIFAR100(data_root, train=True,  download=True, transform=get_transforms(True))
    test_ds    = datasets.CIFAR100(data_root, train=False, download=True, transform=get_transforms(False))

    val_size   = int(0.1 * len(train_full))
    train_size = len(train_full) - val_size
    train_ds, val_ds = random_split(
        train_full, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    # Override val_ds transforms to use eval transforms
    val_eval_ds = datasets.CIFAR100(data_root, train=True, download=False,
                                     transform=get_transforms(False))
    val_ds_final = torch.utils.data.Subset(val_eval_ds, val_ds.indices)

    train_loader = DataLoader(train_ds,     batch_size=BATCH_SIZE, shuffle=True,
                               num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_ds_final, batch_size=BATCH_SIZE, shuffle=False,
                               num_workers=NUM_WORKERS, pin_memory=True)
    test_loader  = DataLoader(test_ds,      batch_size=BATCH_SIZE, shuffle=False,
                               num_workers=NUM_WORKERS, pin_memory=True)
    return train_loader, val_loader, test_loader

CIFAR100_CLASSES = [
    'apple','aquarium_fish','baby','bear','beaver','bed','bee','beetle','bicycle','bottle',
    'bowl','boy','bridge','bus','butterfly','camel','can','castle','caterpillar','cattle',
    'chair','chimpanzee','clock','cloud','cockroach','couch','crab','crocodile','cup',
    'dinosaur','dolphin','elephant','flatfish','forest','fox','girl','hamster','house',
    'kangaroo','keyboard','lamp','lawn_mower','leopard','lion','lizard','lobster','man',
    'maple_tree','motorcycle','mountain','mouse','mushroom','oak_tree','orange','orchid',
    'otter','palm_tree','pear','pickup_truck','pine_tree','plain','plate','poppy',
    'porcupine','possum','rabbit','raccoon','ray','road','rocket','rose','sea','seal',
    'shark','shrew','skunk','skyscraper','snail','snake','spider','squirrel','streetcar',
    'sunflower','sweet_pepper','table','tank','telephone','television','tiger','tractor',
    'train','trout','tulip','turtle','wardrobe','whale','willow_tree','wolf','woman','worm'
]