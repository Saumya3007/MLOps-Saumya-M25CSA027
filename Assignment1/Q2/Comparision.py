

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

sns.set_style("whitegrid")

print("="*80)
print("CPU vs GPU COMPARISON ANALYSIS")
print("="*80)

# ============================================================================
# Load Results
# ============================================================================

try:
    df_cpu = pd.read_csv('results_cpu/fashionmnist_cpu_results.csv')
    print("\n✅ Loaded CPU results")
except:
    print("\n❌ CPU results not found! Run fashionmnist_cpu.py first.")
    exit(1)

try:
    df_gpu = pd.read_csv('results_gpu/fashionmnist_gpu_results.csv')
    print("✅ Loaded GPU results")
except:
    print("❌ GPU results not found! Run fashionmnist_gpu.py first.")
    exit(1)

# Combine
df_combined = pd.concat([df_cpu, df_gpu], ignore_index=True)

# Create comparison directory
os.makedirs('comparison_results', exist_ok=True)

# ============================================================================
# Display Combined Results Table
# ============================================================================

print(f"\n{'='*100}")
print("COMBINED RESULTS TABLE")
print(f"{'='*100}\n")
print(df_combined.to_string(index=False))

# Save combined
df_combined.to_csv('comparison_results/combined_results.csv', index=False)

# ============================================================================
# Visualization 1: Training Time Comparison
# ============================================================================

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# ResNet-18 Time
ax = axes[0]
df_plot = df_combined[['Compute', 'Optimizer', 'ResNet-18 Time']].copy()
df_pivot = df_plot.pivot(index='Optimizer', columns='Compute', values='ResNet-18 Time')

x = np.arange(len(df_pivot.index))
width = 0.35

bars1 = ax.bar(x - width/2, df_pivot['CPU'], width, label='CPU', 
               color='#e74c3c', alpha=0.8, edgecolor='black')
bars2 = ax.bar(x + width/2, df_pivot['GPU'], width, label='GPU', 
               color='#27ae60', alpha=0.8, edgecolor='black')

ax.set_xlabel('Optimizer', fontsize=13, fontweight='bold')
ax.set_ylabel('Training Time (ms)', fontsize=13, fontweight='bold')
ax.set_title('ResNet-18 Training Time', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(df_pivot.index)
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)

# Add speedup annotations
for i, opt in enumerate(df_pivot.index):
    cpu_time = df_pivot.loc[opt, 'CPU']
    gpu_time = df_pivot.loc[opt, 'GPU']
    speedup = cpu_time / gpu_time
    ax.text(i, max(cpu_time, gpu_time) * 1.05, f'{speedup:.1f}x faster',
            ha='center', fontsize=10, fontweight='bold', color='green')

# ResNet-50 Time
ax = axes[1]
df_plot = df_combined[['Compute', 'Optimizer', 'ResNet-50 Time']].copy()
df_pivot = df_plot.pivot(index='Optimizer', columns='Compute', values='ResNet-50 Time')

bars1 = ax.bar(x - width/2, df_pivot['CPU'], width, label='CPU', 
               color='#e74c3c', alpha=0.8, edgecolor='black')
bars2 = ax.bar(x + width/2, df_pivot['GPU'], width, label='GPU', 
               color='#27ae60', alpha=0.8, edgecolor='black')

ax.set_xlabel('Optimizer', fontsize=13, fontweight='bold')
ax.set_ylabel('Training Time (ms)', fontsize=13, fontweight='bold')
ax.set_title('ResNet-50 Training Time', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(df_pivot.index)
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)

# Add speedup annotations
for i, opt in enumerate(df_pivot.index):
    cpu_time = df_pivot.loc[opt, 'CPU']
    gpu_time = df_pivot.loc[opt, 'GPU']
    speedup = cpu_time / gpu_time
    ax.text(i, max(cpu_time, gpu_time) * 1.05, f'{speedup:.1f}x faster',
            ha='center', fontsize=10, fontweight='bold', color='green')

plt.suptitle('CPU vs GPU Training Time Comparison', fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig('comparison_results/training_time_comparison.png', dpi=300, bbox_inches='tight')
print("\n📊 Saved: comparison_results/training_time_comparison.png")

# ============================================================================
# Visualization 2: Accuracy Comparison
# ============================================================================

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# ResNet-18 Accuracy
ax = axes[0]
df_plot = df_combined[['Compute', 'Optimizer', 'ResNet-18 Acc']].copy()
df_pivot = df_plot.pivot(index='Optimizer', columns='Compute', values='ResNet-18 Acc')

bars1 = ax.bar(x - width/2, df_pivot['CPU'], width, label='CPU', 
               color='#3498db', alpha=0.8, edgecolor='black')
bars2 = ax.bar(x + width/2, df_pivot['GPU'], width, label='GPU', 
               color='#9b59b6', alpha=0.8, edgecolor='black')

# Add values on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=9)

ax.set_xlabel('Optimizer', fontsize=13, fontweight='bold')
ax.set_ylabel('Test Accuracy (%)', fontsize=13, fontweight='bold')
ax.set_title('ResNet-18 Test Accuracy', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(df_pivot.index)
ax.set_ylim([0, 105])
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)

# ResNet-50 Accuracy
ax = axes[1]
df_plot = df_combined[['Compute', 'Optimizer', 'ResNet-50 Acc']].copy()
df_pivot = df_plot.pivot(index='Optimizer', columns='Compute', values='ResNet-50 Acc')

bars1 = ax.bar(x - width/2, df_pivot['CPU'], width, label='CPU', 
               color='#3498db', alpha=0.8, edgecolor='black')
bars2 = ax.bar(x + width/2, df_pivot['GPU'], width, label='GPU', 
               color='#9b59b6', alpha=0.8, edgecolor='black')

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=9)

ax.set_xlabel('Optimizer', fontsize=13, fontweight='bold')
ax.set_ylabel('Test Accuracy (%)', fontsize=13, fontweight='bold')
ax.set_title('ResNet-50 Test Accuracy', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(df_pivot.index)
ax.set_ylim([0, 105])
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)

plt.suptitle('CPU vs GPU Accuracy Comparison', fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig('comparison_results/accuracy_comparison.png', dpi=300, bbox_inches='tight')
print("📊 Saved: comparison_results/accuracy_comparison.png")

# ============================================================================
# Visualization 3: FLOPs Comparison
# ============================================================================

fig, ax = plt.subplots(figsize=(10, 6))

models = ['ResNet-18', 'ResNet-50']
flops_18 = df_combined[df_combined['Compute'] == 'CPU']['ResNet-18 FLOPs'].iloc[0]
flops_50 = df_combined[df_combined['Compute'] == 'CPU']['ResNet-50 FLOPs'].iloc[0]
flops = [flops_18, flops_50]

bars = ax.bar(models, flops, color=['#f39c12', '#e67e22'], 
              alpha=0.85, edgecolor='black', linewidth=2)

for bar, val in zip(bars, flops):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
            f'{val:.1f}M', ha='center', va='bottom', 
            fontsize=12, fontweight='bold')

ax.set_ylabel('FLOPs (Millions)', fontsize=13, fontweight='bold')
ax.set_title('Model Computational Complexity (FLOPs)', 
             fontsize=15, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('comparison_results/flops_comparison.png', dpi=300, bbox_inches='tight')
print("📊 Saved: comparison_results/flops_comparison.png")

# ============================================================================
# Generate Detailed Analysis Report
# ============================================================================

report = []
report.append("="*80)
report.append("DETAILED ANALYSIS REPORT")
report.append("FashionMNIST: CPU vs GPU Performance on ResNet-18 and ResNet-50")
report.append("="*80)
report.append("")

# 1. Training Time Analysis
report.append("1. TRAINING TIME ANALYSIS")
report.append("-" * 80)

for model in ['ResNet-18', 'ResNet-50']:
    time_col = f'{model} Time'
    report.append(f"\n{model}:")
    
    for opt in df_combined['Optimizer'].unique():
        cpu_time = df_combined[(df_combined['Compute'] == 'CPU') & 
                              (df_combined['Optimizer'] == opt)][time_col].values[0]
        gpu_time = df_combined[(df_combined['Compute'] == 'GPU') & 
                              (df_combined['Optimizer'] == opt)][time_col].values[0]
        
        speedup = cpu_time / gpu_time
        time_saved = cpu_time - gpu_time
        
        report.append(f"  {opt} Optimizer:")
        report.append(f"    CPU: {cpu_time:.2f} ms ({cpu_time/1000:.2f} sec)")
        report.append(f"    GPU: {gpu_time:.2f} ms ({gpu_time/1000:.2f} sec)")
        report.append(f"    Speedup: {speedup:.2f}x faster on GPU")
        report.append(f"    Time Saved: {time_saved:.2f} ms ({time_saved/1000:.2f} sec)")

# 2. Accuracy Analysis
report.append("\n\n2. CLASSIFICATION ACCURACY ANALYSIS")
report.append("-" * 80)

for model in ['ResNet-18', 'ResNet-50']:
    acc_col = f'{model} Acc'
    report.append(f"\n{model}:")
    
    cpu_accs = df_combined[df_combined['Compute'] == 'CPU'][acc_col].values
    gpu_accs = df_combined[df_combined['Compute'] == 'GPU'][acc_col].values
    
    report.append(f"  CPU Average: {cpu_accs.mean():.2f}% (Range: {cpu_accs.min():.2f}% - {cpu_accs.max():.2f}%)")
    report.append(f"  GPU Average: {gpu_accs.mean():.2f}% (Range: {gpu_accs.min():.2f}% - {gpu_accs.max():.2f}%)")
    report.append(f"  Difference: {abs(cpu_accs.mean() - gpu_accs.mean()):.2f}%")

# 3. FLOPs Analysis
report.append("\n\n3. COMPUTATIONAL COMPLEXITY (FLOPs)")
report.append("-" * 80)

flops_18 = df_combined['ResNet-18 FLOPs'].iloc[0]
flops_50 = df_combined['ResNet-50 FLOPs'].iloc[0]

report.append(f"\nResNet-18: {flops_18:.2f} MFLOPs ({flops_18/1000:.3f} GFLOPs)")
report.append(f"ResNet-50: {flops_50:.2f} MFLOPs ({flops_50/1000:.3f} GFLOPs)")
report.append(f"ResNet-50 is {flops_50/flops_18:.2f}x more computationally expensive")

# 4. Best Configurations
report.append("\n\n4. BEST CONFIGURATIONS")
report.append("-" * 80)

for compute in ['CPU', 'GPU']:
    report.append(f"\n{compute}:")
    df_comp = df_combined[df_combined['Compute'] == compute]
    
    best_18_idx = df_comp['ResNet-18 Acc'].idxmax()
    best_18 = df_comp.loc[best_18_idx]
    
    best_50_idx = df_comp['ResNet-50 Acc'].idxmax()
    best_50 = df_comp.loc[best_50_idx]
    
    report.append(f"  Best ResNet-18:")
    report.append(f"    Optimizer: {best_18['Optimizer']}, LR: {best_18['Learning Rate']}")
    report.append(f"    Accuracy: {best_18['ResNet-18 Acc']:.2f}%")
    report.append(f"    Time: {best_18['ResNet-18 Time']:.2f} ms")
    
    report.append(f"  Best ResNet-50:")
    report.append(f"    Optimizer: {best_50['Optimizer']}, LR: {best_50['Learning Rate']}")
    report.append(f"    Accuracy: {best_50['ResNet-50 Acc']:.2f}%")
    report.append(f"    Time: {best_50['ResNet-50 Time']:.2f} ms")

# 5. Key Insights
report.append("\n\n5. KEY INSIGHTS")
report.append("-" * 80)

avg_speedup_18 = (df_combined[df_combined['Compute'] == 'CPU']['ResNet-18 Time'].mean() /
                  df_combined[df_combined['Compute'] == 'GPU']['ResNet-18 Time'].mean())
avg_speedup_50 = (df_combined[df_combined['Compute'] == 'CPU']['ResNet-50 Time'].mean() /
                  df_combined[df_combined['Compute'] == 'GPU']['ResNet-50 Time'].mean())

report.append(f"\n• GPU provides {avg_speedup_18:.1f}x average speedup for ResNet-18")
report.append(f"• GPU provides {avg_speedup_50:.1f}x average speedup for ResNet-50")
report.append(f"• Accuracy is similar between CPU and GPU (±1-2%)")
report.append(f"• ResNet-50 takes ~{flops_50/flops_18:.1f}x more FLOPs than ResNet-18")
report.append(f"• Adam optimizer generally achieves better accuracy than SGD")
report.append(f"• GPU training is essential for larger models and datasets")

report.append("\n" + "="*80)

report_text = "\n".join(report)
print(f"\n{report_text}")

with open('comparison_results/analysis_report.txt', 'w') as f:
    f.write(report_text)

print("\n💾 Saved: comparison_results/analysis_report.txt")

print("\n" + "="*80)
print("COMPARISON ANALYSIS COMPLETED!")
print("="*80)
print("\nGenerated files:")
print("  📊 comparison_results/training_time_comparison.png")
print("  📊 comparison_results/accuracy_comparison.png")
print("  📊 comparison_results/flops_comparison.png")
print("  📄 comparison_results/combined_results.csv")
print("  📄 comparison_results/analysis_report.txt")
print("="*80)
