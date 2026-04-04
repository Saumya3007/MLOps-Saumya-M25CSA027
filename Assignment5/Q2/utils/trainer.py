import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb


def train_classifier(model, train_loader, test_loader, epochs=100, lr=0.1,
                     device='cuda', run_name='resnet18_cifar10',
                     save_path='./checkpoints/best_resnet18.pth'):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    best_acc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total   += targets.size(0)

        train_acc  = 100.0 * correct / total
        train_loss = running_loss / total
        test_acc   = evaluate(model, test_loader, device)
        scheduler.step()

        wandb.log({'epoch': epoch, 'train_loss': train_loss,
                   'train_acc': train_acc, 'test_acc': test_acc})
        print(f"Epoch {epoch:3d}/{epochs} | Loss: {train_loss:.4f} | "
              f"Train: {train_acc:.2f}% | Test: {test_acc:.2f}%")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), save_path)
            print(f"  -> Saved best model ({best_acc:.2f}%)")

    print(f"\nBest Test Accuracy: {best_acc:.2f}%")
    return best_acc


def evaluate(model, loader, device='cuda'):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            _, predicted = model(inputs).max(1)
            correct += predicted.eq(targets).sum().item()
            total   += targets.size(0)
    return 100.0 * correct / total


def train_detector(model, train_loader, val_loader, epochs=30, lr=1e-3,
                   device='cuda', run_name='detector',
                   save_path='./checkpoints/best_detector.pth'):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    best_acc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total   += targets.size(0)

        train_acc = 100.0 * correct / total
        val_acc   = evaluate(model, val_loader, device)
        scheduler.step()

        wandb.log({f'{run_name}/epoch': epoch,
                   f'{run_name}/train_acc': train_acc,
                   f'{run_name}/val_acc': val_acc})
        print(f"[{run_name}] Epoch {epoch:3d}/{epochs} | "
              f"Train: {train_acc:.2f}% | Val: {val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"  -> Saved best detector ({best_acc:.2f}%)")

    return best_acc