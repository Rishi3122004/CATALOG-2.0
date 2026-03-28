#!/usr/bin/env python
"""
Evaluate CATALOG model on test set and generate confusion matrix
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
from pathlib import Path
import sys
import os

# Import training module to get access to LLaVA_CLIP
sys.path.insert(0, str(Path(__file__).parent))

from models.CATALOG_Projections import LLaVA_CLIP
from utils import BaselineDataset


def main():
    print("="*70)
    print("  CATALOG Model - Test Evaluation & Confusion Matrix")
    print("="*70)
    
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[INFO] Device: {device}")
    
    # Model hyperparameters
    weight_Clip = 0.60855
    num_layers = 4
    dropout = 0.381
    hidden_dim = 1743
    t = 0.1
    
    # Class names for Serengeti dataset
    class_names = [
        'aardvark', 'antelope', 'baboon', 'badger', 'bat',
        'bear', 'bee-eater', 'bird', 'boar', 'buffalo'
    ]
    
    print("\n[CHECK] Loading features...")
    # Load text features
    text_features_path = Path('features/Features_serengeti/standard_features') / 'Text_features_16.pt'
    text_features = torch.load(text_features_path)
    text_features = text_features.float().to(device)
    print(f"  [OK] Text features loaded: {text_features.shape}")
    
    # Load test dataset
    test_dataset_path = Path('features/Features_serengeti/standard_features') / 'Features_CATALOG_test_16.pt'
    test_dataset = BaselineDataset(json_path=test_dataset_path)
    
    print(f"  [OK] Test samples: {len(test_dataset)}")
    
    # Create dataloader
    test_loader = DataLoader(test_dataset, batch_size=26, shuffle=False)
    
    print("\n[CHECK] Loading model checkpoint...")
    # Initialize model
    model = LLaVA_CLIP(
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        device=device
    ).to(device)
    
    # Load checkpoint
    checkpoint_path = Path('Best/exp_Base_In_domain_Serengeti/training_2026-03-28_16-00-22/best_model_params_4_1743.pth')
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        model.load_state_dict(checkpoint)
        print(f"  [OK] Checkpoint loaded from: {checkpoint_path}")
    else:
        print(f"  [ERROR] Checkpoint not found: {checkpoint_path}")
        return
    
    # Evaluate
    print("\n[CHECK] Evaluating on test set...")
    model.eval()
    all_preds = []
    all_targets = []
    correct = 0
    total = 0
    total_loss = 0.0
    
    with torch.no_grad():
        for batch_idx, (image_features, description_embeddings, target) in enumerate(test_loader):
            image_features = image_features.float().to(device)
            description_embeddings = description_embeddings.float().to(device)
            target = target.to(device)
            
            # Forward pass
            loss, acc, predictions = model(
                description_embeddings, 
                image_features, 
                text_features,
                weight_Clip,
                target,
                t
            )
            
            # Calculate accuracy
            total += target.size(0)
            correct += int(acc.item())
            total_loss += loss.item()
            
            # Store predictions and targets
            all_preds.extend(predictions.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            
            if (batch_idx + 1) % 10 == 0:
                print(f'  Batch [{batch_idx+1}/{len(test_loader)}]')
    
    accuracy = 100.0 * correct / total
    avg_loss = total_loss / len(test_loader)
    
    print(f"\nTest Results:")
    print(f"  Accuracy: {accuracy:.4f}%")
    print(f"  Loss:     {avg_loss:.4f}")
    
    # Generate confusion matrix
    print("\n[CHECK] Generating confusion matrix...")
    conf_matrix = confusion_matrix(all_targets, all_preds)
    
    # Create DataFrame for better visualization
    conf_df = pd.DataFrame(
        conf_matrix,
        index=[f'True_{cn}' for cn in class_names],
        columns=[f'Pred_{cn}' for cn in class_names]
    )
    
    # Save confusion matrix
    csv_path = Path('conf_matrix_cis_Base.csv')
    conf_df.to_csv(csv_path)
    print(f"  [OK] Confusion matrix saved to: {csv_path}")
    
    # Print classification report
    print("\n[CHECK] Per-class metrics:")
    print(classification_report(all_targets, all_preds, target_names=class_names, digits=4))
    
    # Summary statistics
    print("\n" + "="*70)
    print("  FINAL RESULTS SUMMARY")
    print("="*70)
    print(f"\nBest Model Checkpoint:")
    print(f"  Path: {checkpoint_path}")
    print(f"  (Saved at epoch 73 during training)")
    print(f"\nValidation Performance (from training):")
    print(f"  Best Val Accuracy: 96.4286%")
    print(f"\nTest Performance:")
    print(f"  Test Accuracy: {accuracy:.4f}%")
    print(f"  Test Loss:    {avg_loss:.4f}")
    print(f"\nConfusion Matrix: {csv_path}")
    print("\n" + "="*70)


if __name__ == '__main__':
    main()
