"""
Q2: FashionMNIST CPU - Save ONLY Best 2 Models (1 per ResNet)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.models import resnet18, resnet50
import pandas as pd
import time
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

try:
    from thop import profile
except:
    os.system('pip install thop -q')
    from thop import profile

device = torch.device('cpu')
print(f"{'='*80}")
print(f"Q2: FashionMNIST on CPU - SAVE BEST MODELS ONLY")
print(f"Running on: {device}")
print(f"{'='*80}")

os.makedirs('results_cpu', exist_ok=True)
os.makedirs('graphs_cpu', exist_ok=True)
os.makedirs('best_cpu_models', exist_ok=True)

# ============================================================================
# Data Preparation
# ============================================================================

transform = transforms.Compose([
    transforms.Resize(32),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

print("\nLoading FashionMNIST...")
full_train = datasets.FashionMNIST(root='./data', train=True, 
                                   transform=transform, download=True)
test_dataset = datasets.FashionMNIST(root='./data', train=False, 
                                     transform=transform, download=True)

total = len(full_train)
train_size = int(0.7 * total)
val_size = int(0.1 * total)
remaining = total - train_size - val_size

train_dataset, val_dataset, _ = random_split(
    full_train, [train_size, val_size, remaining],
    generator=torch.Generator().manual_seed(42)
)

print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

# ============================================================================
# Model and Training Functions
# ============================================================================

def create_model(model_type):
    if model_type == 'resnet18':
        model = resnet18(pretrained=False)
    else:
        model = resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 10)
    return model

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc="Training", leave=False)
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100.*correct/total:.2f}%'})
    
    return running_loss / len(loader), 100. * correct / total

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Validation", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return running_loss / len(loader), 100. * correct / total

def test_model(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Testing", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return 100. * correct / total

# ============================================================================
# Training Function
# ============================================================================

def train_and_log(model_type, batch_size, optimizer_name, lr, epochs=10):
    print(f"\n{'='*80}")
    print(f"Training {model_type} | Batch={batch_size} | Opt={optimizer_name} | LR={lr}")
    print(f"{'='*80}")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    model = create_model(model_type).to(device)
    
    # FLOPs calculation
    dummy_input = torch.randn(1, 3, 32, 32).to(device)
    flops, _ = profile(model, inputs=(dummy_input,), verbose=False)
    flops_millions = flops / 1_000_000
    print(f"💡 FLOPs: {flops_millions:.2f} MFLOPs")
    
    criterion = nn.CrossEntropyLoss()
    
    if optimizer_name == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr)
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    start_time = time.time()
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
    
    total_time = (time.time() - start_time) * 1000
    test_acc = test_model(model, test_loader, device)
    
    print(f"\n✅ Test Accuracy: {test_acc:.2f}%")
    print(f"⏱️  Training Time: {total_time:.2f} ms ({total_time/1000:.2f} sec)")
    
    return {
        'model': model,
        'test_acc': test_acc,
        'train_time_ms': total_time,
        'flops': flops_millions,
        'history': history,
        'config': {
            'batch_size': batch_size,
            'optimizer': optimizer_name,
            'lr': lr,
            'epochs': epochs
        }
    }

# ============================================================================
# Plot Function
# ============================================================================

def plot_history(history, model_type, batch_size, optimizer, lr, compute='CPU'):
    epochs = range(1, len(history['train_loss']) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(epochs, history['train_loss'], 'b-o', label='Train Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-s', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax1.set_title(f'{model_type} - Loss', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    ax2.plot(epochs, history['train_acc'], 'b-o', label='Train Acc', linewidth=2)
    ax2.plot(epochs, history['val_acc'], 'r-s', label='Val Acc', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax2.set_title(f'{model_type} - Accuracy', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.suptitle(f'{compute} | {model_type} | Batch={batch_size} | Opt={optimizer} | LR={lr}',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    filename = f'{model_type}_bs{batch_size}_{optimizer}_lr{lr}.png'
    save_path = os.path.join('graphs_cpu', filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📊 Graph saved: {save_path}")

# ============================================================================
# Run Experiments - Track Best Models
# ============================================================================

configs = [
    {'batch_size': 16, 'optimizer': 'SGD', 'lr': 0.001},
    {'batch_size': 16, 'optimizer': 'Adam', 'lr': 0.001},
]

models = ['resnet18', 'resnet50']
results = []

# Track best models
best_models = {
    'resnet18': {'result': None, 'accuracy': 0},
    'resnet50': {'result': None, 'accuracy': 0}
}

print("\n" + "🔵"*40)
print("STARTING CPU EXPERIMENTS")
print("🔵"*40)

for config in configs:
    for model_type in models:
        result = train_and_log(model_type, config['batch_size'], 
                              config['optimizer'], config['lr'], epochs=10)
        
        plot_history(result['history'], model_type, config['batch_size'], 
                    config['optimizer'], config['lr'], compute='CPU')
        
        results.append({
            'Compute': 'CPU',
            'Batch Size': config['batch_size'],
            'Optimizer': config['optimizer'],
            'Learning Rate': config['lr'],
            'Model': model_type,
            'Test Accuracy (%)': round(result['test_acc'], 2),
            'Train Time (ms)': round(result['train_time_ms'], 2),
            'FLOPs (M)': round(result['flops'], 2)
        })
        
        # Track best model for each ResNet type
        if result['test_acc'] > best_models[model_type]['accuracy']:
            best_models[model_type]['result'] = result
            best_models[model_type]['accuracy'] = result['test_acc']

# ============================================================================
# Save ONLY Best Models (2 models: 1 ResNet-18, 1 ResNet-50)
# ============================================================================

print(f"\n{'='*80}")
print("SAVING BEST MODELS ONLY")
print(f"{'='*80}")

for model_type in ['resnet18', 'resnet50']:
    best = best_models[model_type]
    result = best['result']
    config = result['config']
    
    filename = (f"best_{model_type}_CPU_bs{config['batch_size']}_"
               f"{config['optimizer']}_lr{config['lr']}_"
               f"acc{best['accuracy']:.2f}.pth")
    
    filepath = os.path.join('best_cpu_models', filename)
    
    # Save model checkpoint [web:35][web:38]
    torch.save({
        'model_state_dict': result['model'].state_dict(),
        'test_accuracy': best['accuracy'],
        'config': config,
        'flops': result['flops'],
        'train_time_ms': result['train_time_ms']
    }, filepath)
    
    print(f"\n🏆 Best {model_type.upper()} Saved:")
    print(f"   File: {filename}")
    print(f"   Accuracy: {best['accuracy']:.2f}%")
    print(f"   Config: Batch={config['batch_size']}, Opt={config['optimizer']}, LR={config['lr']}")

# ============================================================================
# Save Results
# ============================================================================

df = pd.DataFrame(results)
df_resnet18 = df[df['Model'] == 'resnet18'][['Compute', 'Batch Size', 'Optimizer', 'Learning Rate', 'Test Accuracy (%)', 'Train Time (ms)', 'FLOPs (M)']].copy()
df_resnet18.columns = ['Compute', 'Batch Size', 'Optimizer', 'Learning Rate', 'ResNet-18 Acc', 'ResNet-18 Time', 'ResNet-18 FLOPs']

df_resnet50 = df[df['Model'] == 'resnet50'][['Test Accuracy (%)', 'Train Time (ms)', 'FLOPs (M)']].copy()
df_resnet50.columns = ['ResNet-50 Acc', 'ResNet-50 Time', 'ResNet-50 FLOPs']

df_final = pd.concat([df_resnet18.reset_index(drop=True), df_resnet50.reset_index(drop=True)], axis=1)

csv_path = 'results_cpu/fashionmnist_cpu_results.csv'
df_final.to_csv(csv_path, index=False)

print(f"\n{'='*80}")
print("CPU RESULTS")
print(f"{'='*80}")
print(df_final.to_string(index=False))

print(f"\n💾 Results: {csv_path}")
print(f"💾 Graphs: graphs_cpu/")
print(f"💾 Best Models (2 only): best_cpu_models/")

print(f"\n{'='*80}")
print("✅ CPU EXPERIMENTS COMPLETED - SAVED ONLY 2 BEST MODELS")
print(f"{'='*80}")
