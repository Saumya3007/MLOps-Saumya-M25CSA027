import torch.nn as nn
from torchvision.models import resnet18, resnet34


def get_resnet18(num_classes=10, pretrained=False):
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def get_resnet34(num_classes=2, pretrained=False):
    model = resnet34(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model