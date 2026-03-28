#!/usr/bin/env python
"""
Comprehensive diagnostic script to check for overfitting and data leakage
"""

import torch
import numpy as np
from pathlib import Path
import json
from collections import Counter

def check_data_leakage():
    """Check if test samples might be in training set"""
    print("\n" + "="*70)
    print("  DATA LEAKAGE DIAGNOSIS")
    print("="*70)
    
    try:
        # Load datasets
        train_data = torch.load('features/Features_serengeti/standard_features/Features_CATALOG_train_16.pt', weights_only=False)
        val_data = torch.load('features/Features_serengeti/standard_features/Features_CATALOG_val_16.pt', weights_only=False)
        test_data = torch.load('features/Features_serengeti/standard_features/Features_CATALOG_test_16.pt', weights_only=False)
        
        # Extract samples
        if isinstance(train_data, dict) and 'target_index' in train_data:
            print("\n[CHECK] Dataset Format: Stacked tensors")
            train_targets = train_data['target_index'].numpy()
            val_targets = val_data['target_index'].numpy()
            test_targets = test_data['target_index'].numpy()
            
            print(f"\nDataset Sizes:")
            print(f"  Train: {len(train_targets)} samples")
            print(f"  Val:   {len(val_targets)} samples")
            print(f"  Test:  {len(test_targets)} samples")
            print(f"  Total: {len(train_targets) + len(val_targets) + len(test_targets)}")
            
            # Check class distribution
            print(f"\n[CHECK] Class Distribution (Train):")
            train_counts = Counter(train_targets)
            for cls_idx in sorted(train_counts.keys()):
                print(f"  Class {cls_idx}: {train_counts[cls_idx]:4d} ({100*train_counts[cls_idx]/len(train_targets):.1f}%)")
            
            print(f"\n[CHECK] Class Distribution (Val):")
            val_counts = Counter(val_targets)
            for cls_idx in sorted(val_counts.keys()):
                print(f"  Class {cls_idx}: {val_counts[cls_idx]:4d} ({100*val_counts[cls_idx]/len(val_targets):.1f}%)")
            
            print(f"\n[CHECK] Class Distribution (Test):")
            test_counts = Counter(test_targets)
            for cls_idx in sorted(test_counts.keys()):
                print(f"  Class {cls_idx}: {test_counts[cls_idx]:4d} ({100*test_counts[cls_idx]/len(test_targets):.1f}%)")
            
            # Check for severe class imbalance
            train_dist = list(train_counts.values())
            if max(train_dist) / min(train_dist) > 2:
                print(f"\n[WARNING] Severe class imbalance detected! Ratio: {max(train_dist)/min(train_dist):.1f}x")
            else:
                print(f"\n[OK] Class distribution balanced (max ratio: {max(train_dist)/min(train_dist):.1f}x)")
        
    except Exception as e:
        print(f"[ERROR] {e}")

def analyze_training_curves():
    """Extract and analyze training curves for overfitting signs"""
    print("\n" + "="*70)
    print("  OVERFITTING DIAGNOSIS")
    print("="*70)
    
    # Parse last training output to reconstruct curves
    output_file = Path('c:/Users/rishi/AppData/Roaming/Code/User/workspaceStorage/5f497553319663daf7565c5457715b9b/GitHub.copilot-chat/chat-session-resources/dacd1853-cf1b-4a5b-85ae-9a2812618182/toolu_bdrk_01PmbZXgL2cUvCKhgL4uzbTL__vscode-1774602343007/content.txt')
    
    try:
        with open(output_file, 'r') as f:
            lines = f.readlines()
        
        epochs_data = []
        for i, line in enumerate(lines):
            if 'Epoch [' in line and 'Train loss' in lines[i+1] if i+1 < len(lines) else False:
                epoch_match = line.strip().split('[')[1].split(']')[0]
                train_line = lines[i+1].strip()
                val_line = lines[i+2].strip() if i+2 < len(lines) else ""
                
                if 'Train loss:' in train_line and 'Val' in val_line:
                    train_loss = float(train_line.split('Train loss: ')[1].split(',')[0])
                    train_acc = float(train_line.split('acc: ')[1])
                    val_acc = float(val_line.split('Val acc: ')[1])
                    val_loss = float(val_line.split('Val loss: ')[1].split(',')[0])
                    
                    epochs_data.append({
                        'epoch': epoch_match,
                        'train_loss': train_loss,
                        'train_acc': train_acc,
                        'val_loss': val_loss,
                        'val_acc': val_acc
                    })
        
        if epochs_data:
            print(f"\n[CHECK] Extracted {len(epochs_data)} epoch records")
            
            # Show key metrics
            early_epochs = epochs_data[:5]
            late_epochs = epochs_data[-5:]
            best_epoch = max(epochs_data, key=lambda x: x['val_acc'])
            
            print(f"\nEarly Training (First 5 epochs):")
            for ep in early_epochs:
                print(f"  Epoch {ep['epoch']}: Train Loss {ep['train_loss']:.4f} → Val Loss {ep['val_loss']:.4f} | Train Acc {ep['train_acc']:.2f}% → Val Acc {ep['val_acc']:.2f}%")
            
            print(f"\nLate Training (Last 5 epochs):")
            for ep in late_epochs:
                print(f"  Epoch {ep['epoch']}: Train Loss {ep['train_loss']:.4f} → Val Loss {ep['val_loss']:.4f} | Train Acc {ep['train_acc']:.2f}% → Val Acc {ep['val_acc']:.2f}%")
            
            print(f"\nBest Validation Epoch:")
            print(f"  Epoch {best_epoch['epoch']}: Train Loss {best_epoch['train_loss']:.4f} | Val Loss {best_epoch['val_loss']:.4f}")
            print(f"  Train Acc {best_epoch['train_acc']:.2f}% | Val Acc {best_epoch['val_acc']:.2f}%")
            
            # Overfitting signals
            print(f"\n[CHECK] Overfitting Signals:")
            avg_early_train_loss = np.mean([e['train_loss'] for e in early_epochs])
            avg_late_train_loss = np.mean([e['train_loss'] for e in late_epochs])
            avg_early_val_loss = np.mean([e['val_loss'] for e in early_epochs])
            avg_late_val_loss = np.mean([e['val_loss'] for e in late_epochs])
            
            train_loss_decrease = (avg_early_train_loss - avg_late_train_loss) / avg_early_train_loss * 100
            val_loss_change = (avg_late_val_loss - avg_early_val_loss) / avg_early_val_loss * 100
            
            print(f"  Train loss decreased by {train_loss_decrease:.1f}%")
            print(f"  Val loss changed by {val_loss_change:+.1f}%")
            
            if val_loss_change > 10:
                print(f"  ⚠️  OVERFITTING DETECTED: Val loss increased significantly in late training")
            elif max(late_epochs, key=lambda x: x['train_acc'])['train_acc'] > best_epoch['val_acc'] + 5:
                print(f"  ⚠️  POSSIBLE OVERFITTING: Train acc >> Val acc gap > 5%")
            else:
                print(f"  ✓ No clear overfitting pattern detected")
            
            # Early stopping analysis
            print(f"\n[CHECK] Model Selection:")
            print(f"  Best Val Accuracy: {best_epoch['val_acc']:.4f}% at epoch {best_epoch['epoch']}")
            print(f"  Final Val Accuracy: {late_epochs[-1]['val_acc']:.4f}% at epoch {late_epochs[-1]['epoch']}")
            gap = best_epoch['val_acc'] - late_epochs[-1]['val_acc']
            if gap > 1:
                print(f"  ⚠️  Early stopping saved {gap:.2f}% accuracy improvement")
        
    except Exception as e:
        print(f"[ERROR] Could not parse training curves: {e}")

def check_feature_quality():
    """Analyze feature statistics"""
    print("\n" + "="*70)
    print("  FEATURE QUALITY ANALYSIS")
    print("="*70)
    
    try:
        train_data = torch.load('features/Features_serengeti/standard_features/Features_CATALOG_train_16.pt', weights_only=False)
        
        if isinstance(train_data, dict) and 'image_features' in train_data:
            img_feats = train_data['image_features']
            desc_feats = train_data['description_embeddings']
            
            print(f"\nImage Features:")
            print(f"  Shape: {img_feats.shape}")
            print(f"  Mean: {img_feats.mean():.6f}")
            print(f"  Std:  {img_feats.std():.6f}")
            print(f"  Min:  {img_feats.min():.6f}")
            print(f"  Max:  {img_feats.max():.6f}")
            
            print(f"\nDescription Features:")
            print(f"  Shape: {desc_feats.shape}")
            print(f"  Mean: {desc_feats.mean():.6f}")
            print(f"  Std:  {desc_feats.std():.6f}")
            print(f"  Min:  {desc_feats.min():.6f}")
            print(f"  Max:  {desc_feats.max():.6f}")
            
            # Check for degeneracies
            img_variance = img_feats.var(dim=0).mean()
            desc_variance = desc_feats.var(dim=0).mean()
            
            print(f"\n[CHECK] Feature Variance (higher = more info):")
            print(f"  Image features per-dim variance: {img_variance:.6f}")
            print(f"  Desc features per-dim variance:  {desc_variance:.6f}")
            
            if img_variance < 0.01:
                print(f"  ⚠️  WARNING: Image features have very low variance - may be degenerate")
            if desc_variance < 0.01:
                print(f"  ⚠️  WARNING: Description features have very low variance - may be degenerate")
    
    except Exception as e:
        print(f"[ERROR] {e}")

def main():
    print("\n" + "█"*70)
    print("█" + " "*68 + "█")
    print("█" + "  CATALOG MODEL DIAGNOSTIC REPORT".center(68) + "█")
    print("█" + " "*68 + "█")
    print("█"*70)
    
    check_data_leakage()
    analyze_training_curves()
    check_feature_quality()
    
    print("\n" + "="*70)
    print("  INTERPRETATION GUIDE")
    print("="*70)
    print("""
Possible Explanations for 98% on 12K vs 90% on 340K:

1. **Pre-trained Feature Strength**: CLIP features are incredibly powerful
   → 12K samples might be enough to achieve high accuracy when using
     frozen CLIP embeddings (already learned on 400M images)

2. **Dataset Difficulty**: Serengeti might be easier than paper's dataset
   → Fewer object categories, better image quality, clearer animals

3. **Different Evaluation Protocol**: Different train/val/test splits
   → The paper might use different evaluation methodology

4. **Actual Overfitting**: Model memorized the training set
   → Look for: Train acc >> Val acc, Val loss increasing, perfect train acc

5. **Class Distribution**: Highly imbalanced classes
   → Few hard classes might be inflating overall accuracy

6. **Data Leakage**: Test samples in training set
   → Check: filename overlap, identical features

Next Steps:
□ If overfitting detected: Use regularization (dropout, L2, data aug)
□ If data leakage found: Rebuild train/val/test splits carefully
□ If features strong: Consider knowledge distillation or harder task
□ If classes imbalanced: Use weighted loss or stratified splitting
""")
    
    print("="*70 + "\n")

if __name__ == '__main__':
    main()
