"""
Q1(a): ResNet-18 and ResNet-50 on MNIST and FashionMNIST
Complete code with model saving and loading functionality
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.models import resnet18, resnet50
import numpy as np
import pandas as pd
from torch.cuda.amp import autocast, GradScaler
import time
from tqdm import tqdm
import os

# Create directories for saving models and results
os.makedirs('saved_models', exist_ok=True)
os.makedirs('results', exist_ok=True)

# Check GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

print("="*100)
print("Q1(a): ResNet EXPERIMENTS ON MNIST AND FASHIONMNIST")
print("="*100)

# ============================================================================
# STEP 1: Data Preparation
# ============================================================================

# Transforms for MNIST/FashionMNIST
transform_train = transforms.Compose([
    transforms.Resize(32),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

transform_test = transforms.Compose([
    transforms.Resize(32),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def prepare_datasets(dataset_name='MNIST', train_ratio=0.7, val_ratio=0.1):
    """
    Load dataset and split into train/val/test
    """
    print(f"\n{'='*80}")
    print(f"Preparing {dataset_name} dataset...")
    print(f"{'='*80}")
    
    if dataset_name == 'MNIST':
        full_dataset = datasets.MNIST(root='./data', train=True, 
                                      transform=transform_train, download=True)
        test_dataset_orig = datasets.MNIST(root='./data', train=False, 
                                           transform=transform_test, download=True)
    else:  # FashionMNIST
        full_dataset = datasets.FashionMNIST(root='./data', train=True, 
                                             transform=transform_train, download=True)
        test_dataset_orig = datasets.FashionMNIST(root='./data', train=False, 
                                                  transform=transform_test, download=True)
    
    total_size = len(full_dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    remaining = total_size - train_size - val_size
    
    train_dataset, val_dataset, _ = random_split(
        full_dataset, 
        [train_size, val_size, remaining],
        generator=torch.Generator().manual_seed(42)
    )
    
    test_dataset = test_dataset_orig
    
    print(f"✅ {dataset_name} Split:")
    print(f"   Train: {len(train_dataset)} samples")
    print(f"   Val: {len(val_dataset)} samples")
    print(f"   Test: {len(test_dataset)} samples")
    
    return train_dataset, val_dataset, test_dataset

# Prepare both datasets
print("\n" + "🔵"*40)
print("LOADING DATASETS")
print("🔵"*40)

train_mnist, val_mnist, test_mnist = prepare_datasets('MNIST')
train_fashion, val_fashion, test_fashion = prepare_datasets('FashionMNIST')

# ============================================================================
# STEP 2: Model Creation
# ============================================================================

def create_resnet_model(model_type='resnet18', num_classes=10):
    """
    Create ResNet model
    """
    if model_type == 'resnet18':
        model = resnet18(pretrained=False)
    else:  # resnet50
        model = resnet50(pretrained=False)
    
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    
    return model

# ============================================================================
# STEP 3: Training Functions
# ============================================================================

def train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, use_amp=True):
    """
    Train for one epoch
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc="Training", leave=False)
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        with autocast(enabled=use_amp):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 
                         'acc': f'{100.*correct/total:.2f}%'})
    
    train_loss = running_loss / len(train_loader)
    train_acc = 100. * correct / total
    
    return train_loss, train_acc

def validate(model, val_loader, criterion, device):
    """
    Validate the model
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validation", leave=False)
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 
                             'acc': f'{100.*correct/total:.2f}%'})
    
    val_loss = running_loss / len(val_loader)
    val_acc = 100. * correct / total
    
    return val_loss, val_acc

def test(model, test_loader, device):
    """
    Test the model
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Testing", leave=False)
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({'acc': f'{100.*correct/total:.2f}%'})
    
    test_acc = 100. * correct / total
    return test_acc

# ============================================================================
# STEP 4: Save/Load Model Functions
# ============================================================================

def save_model(model, model_type, dataset_name, batch_size, optimizer_name, 
               learning_rate, epochs, pin_memory, test_acc):
    """
    Save trained model with detailed naming
    """
    filename = f"{model_type}_{dataset_name}_bs{batch_size}_{optimizer_name}_lr{learning_rate}_ep{epochs}_pm{pin_memory}_acc{test_acc:.2f}.pth"
    filepath = os.path.join('saved_models', filename)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_type': model_type,
        'dataset': dataset_name,
        'batch_size': batch_size,
        'optimizer': optimizer_name,
        'learning_rate': learning_rate,
        'epochs': epochs,
        'pin_memory': pin_memory,
        'test_accuracy': test_acc
    }, filepath)
    
    print(f"💾 Model saved: {filename}")
    return filepath

def load_model(filepath, device):
    """
    Load a saved model
    """
    checkpoint = torch.load(filepath, map_location=device)
    
    model = create_resnet_model(checkpoint['model_type'], num_classes=10)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    print(f"✅ Model loaded from: {filepath}")
    print(f"   Test Accuracy: {checkpoint['test_accuracy']:.2f}%")
    
    return model, checkpoint

def check_if_model_exists(model_type, dataset_name, batch_size, optimizer_name, 
                          learning_rate, epochs, pin_memory):
    """
    Check if a model with these exact parameters already exists
    """
    pattern = f"{model_type}_{dataset_name}_bs{batch_size}_{optimizer_name}_lr{learning_rate}_ep{epochs}_pm{pin_memory}"
    
    for filename in os.listdir('saved_models'):
        if filename.startswith(pattern) and filename.endswith('.pth'):
            return os.path.join('saved_models', filename)
    
    return None

# ============================================================================
# STEP 5: Main Training Pipeline with Save/Load
# ============================================================================

def train_model(model_type, dataset_name, batch_size, optimizer_name, 
                learning_rate, num_epochs, pin_memory, use_amp=True, 
                force_retrain=False):
    """
    Complete training pipeline with model saving/loading
    """
    # Check if model already exists
    if not force_retrain:
        existing_model = check_if_model_exists(
            model_type, dataset_name, batch_size, optimizer_name,
            learning_rate, num_epochs, pin_memory
        )
        
        if existing_model:
            print(f"\n{'='*80}")
            print(f"⚡ FOUND EXISTING MODEL")
            print(f"{'='*80}")
            model, checkpoint = load_model(existing_model, device)
            print(f"Skipping training, using saved accuracy: {checkpoint['test_accuracy']:.2f}%")
            return checkpoint['test_accuracy'], 0, existing_model
    
    print(f"\n{'='*80}")
    print(f"Training {model_type} on {dataset_name}")
    print(f"Batch: {batch_size}, Optimizer: {optimizer_name}, LR: {learning_rate}")
    print(f"Epochs: {num_epochs}, Pin Memory: {pin_memory}, AMP: {use_amp}")
    print(f"{'='*80}")
    
    # Prepare data
    if dataset_name == 'MNIST':
        train_dataset, val_dataset, test_dataset = train_mnist, val_mnist, test_mnist
    else:
        train_dataset, val_dataset, test_dataset = train_fashion, val_fashion, test_fashion
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                             shuffle=True, pin_memory=pin_memory, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                           shuffle=False, pin_memory=pin_memory, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                            shuffle=False, pin_memory=pin_memory, num_workers=2)
    
    # Create model
    model = create_resnet_model(model_type, num_classes=10).to(device)
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    if optimizer_name == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    else:  # Adam
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Gradient scaler for AMP
    scaler = GradScaler(enabled=use_amp)
    
    # Training loop
    best_val_acc = 0.0
    start_time = time.time()
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Train
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, 
                                                optimizer, scaler, device, use_amp)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
    
    # Test on test set
    test_acc = test(model, test_loader, device)
    training_time = time.time() - start_time
    
    print(f"\n{'='*80}")
    print(f"✅ TRAINING COMPLETED")
    print(f"Final Test Accuracy: {test_acc:.2f}%")
    print(f"Training Time: {training_time/60:.2f} minutes")
    print(f"{'='*80}")
    
    # Save the model
    model_path = save_model(model, model_type, dataset_name, batch_size, 
                           optimizer_name, learning_rate, num_epochs, 
                           pin_memory, test_acc)
    
    # Clear GPU cache
    del model
    torch.cuda.empty_cache()
    
    return test_acc, training_time, model_path

# ============================================================================
# STEP 6: Define Experiment Configurations
# ============================================================================

# Hyperparameter grid
batch_sizes = [16, 32]
optimizers = ['SGD', 'Adam']
learning_rates = [0.001, 0.0001]
pin_memory_options = [True]  # Using True for better performance
epochs_options = [2, 3]  # Two different epoch settings

# Dataset names
datasets = ['MNIST', 'FashionMNIST']

print(f"\n{'='*80}")
print(f"EXPERIMENT CONFIGURATION")
print(f"{'='*80}")
print(f"Datasets: {datasets}")
print(f"Batch sizes: {batch_sizes}")
print(f"Optimizers: {optimizers}")
print(f"Learning rates: {learning_rates}")
print(f"Epochs options: {epochs_options}")
print(f"Pin memory: {pin_memory_options}")
print(f"Total experiments per model: {len(datasets) * len(batch_sizes) * len(optimizers) * len(learning_rates) * len(epochs_options) * len(pin_memory_options)}")
print(f"Total models to train: {2 * len(datasets) * len(batch_sizes) * len(optimizers) * len(learning_rates) * len(epochs_options) * len(pin_memory_options)}")
print(f"{'='*80}")

# ============================================================================
# STEP 7: Run All Experiments
# ============================================================================

# Storage for results
all_results = []

# Run experiments
print("\n" + "🚀"*40)
print("STARTING ALL RESNET EXPERIMENTS")
print("🚀"*40)

experiment_count = 0
total_experiments = len(datasets) * len(batch_sizes) * len(optimizers) * len(learning_rates) * len(epochs_options) * len(pin_memory_options)

for dataset in datasets:
    print(f"\n{'#'*100}")
    print(f"# DATASET: {dataset}")
    print(f"{'#'*100}")
    
    for epochs in epochs_options:
        for pin_mem in pin_memory_options:
            for batch_size in batch_sizes:
                for optimizer in optimizers:
                    for lr in learning_rates:
                        experiment_count += 1
                        print(f"\n{'─'*100}")
                        print(f"Experiment {experiment_count}/{total_experiments} (per model)")
                        print(f"{'─'*100}")
                        
                        # Train ResNet-18
                        acc_18, time_18, path_18 = train_model(
                            'resnet18', dataset, batch_size, optimizer, 
                            lr, epochs, pin_mem, use_amp=True, 
                            force_retrain=False  # Set to True to retrain existing models
                        )
                        
                        # Train ResNet-50
                        acc_50, time_50, path_50 = train_model(
                            'resnet50', dataset, batch_size, optimizer, 
                            lr, epochs, pin_mem, use_amp=True,
                            force_retrain=False
                        )
                        
                        # Store results
                        all_results.append({
                            'Dataset': dataset,
                            'Batch Size': batch_size,
                            'Optimizer': optimizer,
                            'Learning Rate': lr,
                            'Epochs': epochs,
                            'Pin Memory': pin_mem,
                            'ResNet-18 Acc (%)': round(acc_18, 2),
                            'ResNet-50 Acc (%)': round(acc_50, 2),
                            'ResNet-18 Time (min)': round(time_18/60, 2) if time_18 > 0 else 0,
                            'ResNet-50 Time (min)': round(time_50/60, 2) if time_50 > 0 else 0,
                            'ResNet-18 Model': path_18,
                            'ResNet-50 Model': path_50
                        })

print("\n" + "✅"*40)
print("ALL EXPERIMENTS COMPLETED!")
print("✅"*40)

# ============================================================================
# STEP 8: Save and Display Results
# ============================================================================

# Create DataFrame
df_results = pd.DataFrame(all_results)

# Save complete results
complete_csv = 'results/resnet_complete_results.csv'
df_results.to_csv(complete_csv, index=False)
print(f"\n💾 Complete results saved to: {complete_csv}")

# Save separate files for each dataset
for dataset in datasets:
    df_dataset = df_results[df_results['Dataset'] == dataset]
    dataset_csv = f'results/resnet_{dataset.lower()}_results.csv'
    df_dataset.to_csv(dataset_csv, index=False)
    print(f"💾 {dataset} results saved to: {dataset_csv}")

# Display results for each dataset
for dataset in datasets:
    print(f"\n{'='*100}")
    print(f"{dataset} RESULTS")
    print(f"{'='*100}")
    df_dataset = df_results[df_results['Dataset'] == dataset]
    print(df_dataset[['Batch Size', 'Optimizer', 'Learning Rate', 'Epochs',
                      'ResNet-18 Acc (%)', 'ResNet-50 Acc (%)']].to_string(index=False))

# ============================================================================
# STEP 9: Analysis
# ============================================================================

print(f"\n{'='*100}")
print(f"ANALYSIS")
print(f"{'='*100}")

for dataset in datasets:
    df_dataset = df_results[df_results['Dataset'] == dataset]
    
    print(f"\n{'─'*80}")
    print(f"{dataset} Dataset")
    print(f"{'─'*80}")
    
    # Best ResNet-18
    best_18 = df_dataset.loc[df_dataset['ResNet-18 Acc (%)'].idxmax()]
    print(f"\n🏆 Best ResNet-18:")
    print(f"   Accuracy: {best_18['ResNet-18 Acc (%)']}%")
    print(f"   Config: Batch={best_18['Batch Size']}, Optimizer={best_18['Optimizer']}, LR={best_18['Learning Rate']}, Epochs={best_18['Epochs']}")
    
    # Best ResNet-50
    best_50 = df_dataset.loc[df_dataset['ResNet-50 Acc (%)'].idxmax()]
    print(f"\n🏆 Best ResNet-50:")
    print(f"   Accuracy: {best_50['ResNet-50 Acc (%)']}%")
    print(f"   Config: Batch={best_50['Batch Size']}, Optimizer={best_50['Optimizer']}, LR={best_50['Learning Rate']}, Epochs={best_50['Epochs']}")
    
    # Average performance
    print(f"\n📊 Average Performance:")
    print(f"   ResNet-18: {df_dataset['ResNet-18 Acc (%)'].mean():.2f}%")
    print(f"   ResNet-50: {df_dataset['ResNet-50 Acc (%)'].mean():.2f}%")

# Compare datasets
print(f"\n{'─'*80}")
print(f"Cross-Dataset Comparison")
print(f"{'─'*80}")

for model in ['ResNet-18 Acc (%)', 'ResNet-50 Acc (%)']:
    print(f"\n{model}:")
    for dataset in datasets:
        df_dataset = df_results[df_results['Dataset'] == dataset]
        print(f"   {dataset}: Best={df_dataset[model].max():.2f}%, Avg={df_dataset[model].mean():.2f}%")

# ============================================================================
# STEP 10: Create Summary for Assignment Submission
# ============================================================================

print(f"\n{'='*100}")
print(f"ASSIGNMENT TABLE FORMAT")
print(f"{'='*100}")

# Create main table for assignment (using best epoch setting)
for dataset in datasets:
    print(f"\n{dataset} - Test Classification Accuracy (%)")
    print(f"{'─'*100}")
    
    df_dataset = df_results[(df_results['Dataset'] == dataset) & 
                           (df_results['Epochs'] == 10) &
                           (df_results['Pin Memory'] == True)]
    
    print(f"{'Batch Size':<12} {'Optimizer':<12} {'Learning Rate':<15} {'ResNet-18':<12} {'ResNet-50':<12}")
    print(f"{'─'*100}")
    
    for _, row in df_dataset.iterrows():
        print(f"{row['Batch Size']:<12} {row['Optimizer']:<12} {row['Learning Rate']:<15} {row['ResNet-18 Acc (%)']:<12} {row['ResNet-50 Acc (%)']:<12}")

print(f"\n{'='*100}")
print(f"ALL RESULTS AND MODELS SAVED!")
print(f"{'='*100}")
print(f"\nSaved files:")
print(f"  📁 Models directory: saved_models/")
print(f"  📁 Results directory: results/")
print(f"  📄 Complete results: {complete_csv}")
for dataset in datasets:
    print(f"  📄 {dataset} results: results/resnet_{dataset.lower()}_results.csv")
print(f"{'='*100}")
