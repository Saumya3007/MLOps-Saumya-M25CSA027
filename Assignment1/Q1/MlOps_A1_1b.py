import numpy as np
import pandas as pd
import time
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import joblib
import os

print("="*100)
print("Q1(b): SVM EXPERIMENTS ")
print("="*100)


os.makedirs('best_svm_models', exist_ok=True)

# ============================================================================
# Data Preparation
# ============================================================================

def prepare_svm_data(dataset_name='MNIST'):
    print(f"\n{'='*80}")
    print(f"Loading {dataset_name} dataset...")
    print(f"{'='*80}")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    if dataset_name == 'MNIST':
        full_train = datasets.MNIST(root='./data', train=True, 
                                    transform=transform, download=True)
        test_dataset = datasets.MNIST(root='./data', train=False, 
                                     transform=transform, download=True)
    else:
        full_train = datasets.FashionMNIST(root='./data', train=True, 
                                          transform=transform, download=True)
        test_dataset = datasets.FashionMNIST(root='./data', train=False, 
                                            transform=transform, download=True)
    
    total_size = len(full_train)
    train_size = int(0.7 * total_size)
    val_size = int(0.1 * total_size)
    remaining = total_size - train_size - val_size
    
    train_dataset, val_dataset, _ = random_split(
        full_train, 
        [train_size, val_size, remaining],
        generator=torch.Generator().manual_seed(42)
    )
    
    def dataset_to_numpy(dataset):
        loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
        data, labels = next(iter(loader))
        data_flat = data.view(data.size(0), -1).numpy()
        labels_np = labels.numpy()
        return data_flat, labels_np
    
    X_train, y_train = dataset_to_numpy(train_dataset)
    X_val, y_val = dataset_to_numpy(val_dataset)
    X_test, y_test = dataset_to_numpy(test_dataset)
    
    print(f"✅ Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test

def scale_features(X_train, X_val, X_test):
    print("\n🔄 Scaling features...")
    scaler = StandardScaler()
    scaler.fit(X_train)
    
    X_train_scaled = scaler.transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    print("✅ Feature scaling completed!")
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler

# ============================================================================
# Training Function with Train Accuracy
# ============================================================================

def train_svm(X_train, y_train, X_test, y_test, kernel='rbf', C=1.0, 
              gamma='scale', degree=3, dataset_name='MNIST'):
    """
    Train SVM and return both train and test accuracy
    """
    print(f"\n{'='*80}")
    print(f"Training SVM on {dataset_name}")
    print(f"Kernel: {kernel}, C: {C}, Gamma: {gamma}", end='')
    if kernel == 'poly':
        print(f", Degree: {degree}")
    else:
        print()
    print(f"{'='*80}")
    
    if kernel == 'poly':
        svm_model = SVC(kernel=kernel, C=C, gamma=gamma, degree=degree, 
                       cache_size=2000, verbose=False, max_iter=-1)
    else:
        svm_model = SVC(kernel=kernel, C=C, gamma=gamma, 
                       cache_size=2000, verbose=False, max_iter=-1)
    
    print("⏳ Training...")
    start_time = time.time()
    svm_model.fit(X_train, y_train)
    training_time = (time.time() - start_time) * 1000
    
    # Train accuracy
    print("📊 Computing train accuracy...")
    y_train_pred = svm_model.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred) * 100
    
    # Test accuracy
    print("🔍 Computing test accuracy...")
    test_start = time.time()
    y_pred = svm_model.predict(X_test)
    test_time = (time.time() - test_start) * 1000
    test_accuracy = accuracy_score(y_test, y_pred) * 100
    
    print(f"\n📊 Results:")
    print(f"   Train Accuracy: {train_accuracy:.2f}%")
    print(f"   Test Accuracy: {test_accuracy:.2f}%")
    print(f"   Training Time: {training_time:.2f} ms ({training_time/1000:.2f} sec)")
    print(f"{'='*80}\n")
    
    return train_accuracy, test_accuracy, training_time, test_time, svm_model

# ============================================================================
# SVM Configurations
# ============================================================================

svm_configs = [
    {'kernel': 'rbf', 'C': 1.0, 'gamma': 'scale'},
    {'kernel': 'rbf', 'C': 10.0, 'gamma': 'scale'},
    {'kernel': 'rbf', 'C': 1.0, 'gamma': 0.001},
    {'kernel': 'rbf', 'C': 10.0, 'gamma': 0.001},
    {'kernel': 'rbf', 'C': 100.0, 'gamma': 'scale'},
    {'kernel': 'rbf', 'C': 0.1, 'gamma': 'scale'},
    {'kernel': 'poly', 'C': 1.0, 'gamma': 'scale', 'degree': 2},
    {'kernel': 'poly', 'C': 1.0, 'gamma': 'scale', 'degree': 3},
    {'kernel': 'poly', 'C': 10.0, 'gamma': 'scale', 'degree': 2},
    {'kernel': 'poly', 'C': 10.0, 'gamma': 'scale', 'degree': 3},
    {'kernel': 'poly', 'C': 1.0, 'gamma': 0.01, 'degree': 3},
    {'kernel': 'poly', 'C': 0.1, 'gamma': 'scale', 'degree': 2},
]

# ============================================================================
# Load and Prepare Datasets
# ============================================================================

# MNIST
X_train_mnist, y_train_mnist, X_val_mnist, y_val_mnist, X_test_mnist, y_test_mnist = \
    prepare_svm_data('MNIST')
X_train_mnist_scaled, X_val_mnist_scaled, X_test_mnist_scaled, scaler_mnist = \
    scale_features(X_train_mnist, X_val_mnist, X_test_mnist)

# FashionMNIST
X_train_fashion, y_train_fashion, X_val_fashion, y_val_fashion, X_test_fashion, y_test_fashion = \
    prepare_svm_data('FashionMNIST')
X_train_fashion_scaled, X_val_fashion_scaled, X_test_fashion_scaled, scaler_fashion = \
    scale_features(X_train_fashion, X_val_fashion, X_test_fashion)

# ============================================================================
# Run Experiments - Track Best Models
# ============================================================================

svm_results = []
best_models = {
    'MNIST': {'model': None, 'accuracy': 0, 'config': None, 'scaler': None},
    'FashionMNIST': {'model': None, 'accuracy': 0, 'config': None, 'scaler': None}
}

print("\n" + "🔵"*40)
print("STARTING SVM EXPERIMENTS")
print("🔵"*40)

# MNIST Experiments
for i, config in enumerate(svm_configs, 1):
    print(f"\n📌 MNIST - Experiment {i}/{len(svm_configs)}")
    
    train_acc, test_acc, train_time, test_time, model = train_svm(
        X_train_mnist_scaled, 
        y_train_mnist,
        X_test_mnist_scaled,
        y_test_mnist,
        dataset_name='MNIST',
        **config
    )
    
    svm_results.append({
        'Dataset': 'MNIST',
        'Kernel': config['kernel'],
        'C': config['C'],
        'Gamma': config['gamma'],
        'Degree': config.get('degree', '-'),
        'Train Accuracy (%)': round(train_acc, 2),
        'Test Accuracy (%)': round(test_acc, 2),
        'Training Time (ms)': round(train_time, 2),
        'Training Time (sec)': round(train_time/1000, 2),
        'Test Time (ms)': round(test_time, 2)
    })
    
    # Track best model for MNIST
    if test_acc > best_models['MNIST']['accuracy']:
        best_models['MNIST']['model'] = model
        best_models['MNIST']['accuracy'] = test_acc
        best_models['MNIST']['config'] = config.copy()
        best_models['MNIST']['scaler'] = scaler_mnist

# FashionMNIST Experiments
for i, config in enumerate(svm_configs, 1):
    print(f"\n📌 FashionMNIST - Experiment {i}/{len(svm_configs)}")
    
    train_acc, test_acc, train_time, test_time, model = train_svm(
        X_train_fashion_scaled, 
        y_train_fashion,
        X_test_fashion_scaled,
        y_test_fashion,
        dataset_name='FashionMNIST',
        **config
    )
    
    svm_results.append({
        'Dataset': 'FashionMNIST',
        'Kernel': config['kernel'],
        'C': config['C'],
        'Gamma': config['gamma'],
        'Degree': config.get('degree', '-'),
        'Train Accuracy (%)': round(train_acc, 2),
        'Test Accuracy (%)': round(test_acc, 2),
        'Training Time (ms)': round(train_time, 2),
        'Training Time (sec)': round(train_time/1000, 2),
        'Test Time (ms)': round(test_time, 2)
    })
    
    # Track best model for FashionMNIST
    if test_acc > best_models['FashionMNIST']['accuracy']:
        best_models['FashionMNIST']['model'] = model
        best_models['FashionMNIST']['accuracy'] = test_acc
        best_models['FashionMNIST']['config'] = config.copy()
        best_models['FashionMNIST']['scaler'] = scaler_fashion

print("\n" + "✅"*40)
print("ALL EXPERIMENTS COMPLETED!")
print("✅"*40)

# ============================================================================
# Save ONLY Best Models
# ============================================================================

print(f"\n{'='*80}")
print("SAVING BEST MODELS ONLY")
print(f"{'='*80}")

for dataset in ['MNIST', 'FashionMNIST']:
    best = best_models[dataset]
    config = best['config']
    
    kernel = config['kernel']
    C = config['C']
    gamma = str(config['gamma']).replace('.', '_')
    degree = config.get('degree', '')
    
    if kernel == 'poly':
        filename = f"best_svm_{dataset}_{kernel}_C{C}_gamma{gamma}_deg{degree}_acc{best['accuracy']:.2f}.pkl"
    else:
        filename = f"best_svm_{dataset}_{kernel}_C{C}_gamma{gamma}_acc{best['accuracy']:.2f}.pkl"
    
    filepath = os.path.join('best_svm_models', filename)
    
 
    save_dict = {
        'model': best['model'],
        'scaler': best['scaler'],
        'config': config,
        'accuracy': best['accuracy'],
        'dataset': dataset
    }
    
    joblib.dump(save_dict, filepath)
    
    print(f"\n🏆 Best {dataset} Model Saved:")
    print(f"   File: {filename}")
    print(f"   Accuracy: {best['accuracy']:.2f}%")
    print(f"   Config: {config}")

# ============================================================================
# Save Results CSV
# ============================================================================

df_svm = pd.DataFrame(svm_results)

print(f"\n{'='*80}")
print("SVM RESULTS")
print(f"{'='*80}")
print(df_svm.to_string(index=False))

csv_filename = 'svm_results.csv'
df_svm.to_csv(csv_filename, index=False)
print(f"\n💾 Results saved to '{csv_filename}'")

# ============================================================================
# Display Best Models Summary
# ============================================================================

print(f"\n{'='*80}")
print("BEST MODELS SUMMARY")
print(f"{'='*80}")

for dataset in ['MNIST', 'FashionMNIST']:
    best = best_models[dataset]
    print(f"\n🏆 {dataset}:")
    print(f"   Accuracy: {best['accuracy']:.2f}%")
    print(f"   Kernel: {best['config']['kernel']}")
    print(f"   C: {best['config']['C']}")
    print(f"   Gamma: {best['config']['gamma']}")
    if 'degree' in best['config']:
        print(f"   Degree: {best['config']['degree']}")

print(f"\n{'='*80}")
print("✅ SAVED ONLY 2 BEST MODELS (1 per dataset)")
print("📁 Location: best_svm_models/")
print(f"{'='*80}")
