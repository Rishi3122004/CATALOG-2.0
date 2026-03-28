"""
Complete pipeline: Extract features from new dataset and train improved CATALOG model
Steps:
1. Extract CLIP + BERT features from new 10-class balanced wilddata
2. Train CATALOG model with best hyperparameters from previous ablation
"""

import os
import sys
import subprocess
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, r'C:\Users\rishi\CATALOG')

from models import CATALOG_Base as md
from train.Base.Train_CATALOG_Base_out_domain import Train_LLaVA_CLIP
import utils

def run_feature_extraction():
    """Step 1: Extract features from new dataset"""
    print("\n" + "="*70)
    print("STEP 1: EXTRACTING FEATURES FROM NEW BALANCED DATASET")
    print("="*70)
    
    extraction_script = r'C:\Users\rishi\CATALOG\feature_extraction\Base\CATALOG_extraction_features_serengeti.py'
    
    if not os.path.exists(extraction_script):
        print(f"❌ Feature extraction script not found: {extraction_script}")
        return False
    
    print("\nRunning feature extraction...")
    result = subprocess.run(
        [r'C:\Users\rishi\anaconda3\envs\CATALOG\python.exe', extraction_script],
        capture_output=False,
        text=True
    )
    
    if result.returncode == 0:
        print("✓ Feature extraction complete!")
        return True
    else:
        print("❌ Feature extraction failed!")
        return False

def run_training():
    """Step 2: Train model with best ablation hyperparameters"""
    print("\n" + "="*70)
    print("STEP 2: TRAINING IMPROVED CATALOG MODEL")
    print("="*70)
    
    # Best hyperparameters from previous ablation search (fusion_more_bert config)
    best_config = {
        'hidden_dim': 1045,
        'num_layers': 1,
        'dropout': 0.27822,
        'lr': 0.03,
        'weight_p': 0.45,  # 45% CLIP, 55% BERT
        'temperature': 0.2,
        'enable_classifier_fusion': True,
        'fusion_init': -2.2
    }
    
    print("\nTraining Configuration (from best ablation trial):")
    for key, value in best_config.items():
        print(f"  {key}: {value}")
    
    # Initialize trainer
    model_kwargs = {
        'enable_classifier_fusion': best_config['enable_classifier_fusion'],
        'fusion_init': best_config['fusion_init']
    }
    
    trainer = Train_LLaVA_CLIP(
        hidden_dim=best_config['hidden_dim'],
        num_layers=best_config['num_layers'],
        dropout=best_config['dropout'],
        lr=best_config['lr'],
        weight_p=best_config['weight_p'],
        temperature=best_config['temperature'],
        num_epochs=20,
        patience=7,
        batch_size=48,
        model_kwargs=model_kwargs
    )
    
    print("\nStarting training...")
    metrics = trainer.train()
    
    if metrics:
        print("\n✓ Training complete!")
        print(f"  Best validation accuracy: {metrics.get('best_val_acc', 'N/A'):.4f}")
        print(f"  Test accuracy: {metrics.get('final_cis_test_acc', 'N/A'):.4f}")
        print(f"  Model saved to: {metrics.get('model_params_path', 'N/A')}")
        return True
    else:
        print("❌ Training failed!")
        return False

def main():
    print("\n" + "="*70)
    print("NEW DATASET PIPELINE: Feature Extraction + Training")
    print("="*70)
    
    # Step 1: Feature Extraction
    if not run_feature_extraction():
        print("\n❌ Pipeline failed at feature extraction")
        sys.exit(1)
    
    # Step 2: Training
    if not run_training():
        print("\n❌ Pipeline failed at training")
        sys.exit(1)
    
    print("\n" + "="*70)
    print("✓ PIPELINE COMPLETE!")
    print("="*70)
    print("\nSummary:")
    print("  ✓ Features extracted from new 10-class balanced dataset")
    print("  ✓ Model trained with improved CATALOG architecture")
    print("  ✓ Fusion head + BERT weighting optimized for better accuracy")
    print("\nNext steps:")
    print("  1. Check test accuracy improvements over old Serengeti baseline (45.67%)")
    print("  2. Compare confusion matrix with previous ablation results")
    print("  3. Potentially further tune hyperparameters if needed")
    print("="*70)

if __name__ == "__main__":
    main()
