"""
Direct CATALOG Base Model Training Script
Uses pre-extracted features - no dataset needed
"""

import os
import sys
import torch
from pathlib import Path

# Ensure we're in the correct directory
script_dir = Path(__file__).parent
os.chdir(script_dir)
sys.path.insert(0, str(script_dir))

from models import CATALOG_Base as base
from utils import BaselineDataset, dataloader_baseline, build_optimizer
from train.Base.Train_CATALOG_Base_out_domain import CATALOG_base


def main():
    print("\n" + "=" * 70)
    print("  CATALOG Base Model - Training with Pre-extracted Features")
    print("=" * 70)
    
    # Check GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"\n✓ GPU Available: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("\n⚠ GPU NOT available - using CPU")
    
    # Feature paths
    features = {
        "train": "features/Features_serengeti/standard_features/Features_CATALOG_train_16.pt",
        "val": "features/Features_serengeti/standard_features/Features_CATALOG_val_16.pt",
        "test_cis": "features/Features_terra/standard_features/Features_CATALOG_cis_test_16.pt",
        "test_trans": "features/Features_terra/standard_features/Features_CATALOG_trans_test_16.pt",
        "text_serengeti": "features/Features_serengeti/standard_features/Text_features_16.pt",
        "text_terra": "features/Features_terra/standard_features/Text_features_16.pt",
    }
    
    # Check all features exist
    print("\n✓ Verifying feature files...")
    all_exist = True
    for name, path in features.items():
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / (1024 ** 2)
            print(f"  ✓ {name:20} ({size_mb:7.1f} MB)")
        else:
            print(f"  ✗ {name:20} MISSING!")
            all_exist = False
    
    if not all_exist:
        print("\n❌ Some feature files are missing. Please run:")
        print("   python extract_features_fixed.py")
        sys.exit(1)
    
    print("\n" + "-" * 70)
    print("Training Configuration:")
    print("-" * 70)
    
    # Training hyperparameters
    config = {
        "weight_Clip": 0.6,
        "num_epochs": 50,
        "batch_size": 48,
        "num_layers": 1,
        "dropout": 0.27822,
        "hidden_dim": 1045,
        "lr": 0.07641,
        "t": 0.1,
        "momentum": 0.8409,
        "patience": 5,
    }
    
    for key, value in config.items():
        print(f"  {key:20}: {value}")
    
    print("\n" + "-" * 70)
    print("Starting training...\n")
    
    # Initialize model
    model = CATALOG_base(
        weight_Clip=config["weight_Clip"],
        num_epochs=config["num_epochs"],
        batch_size=config["batch_size"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
        hidden_dim=config["hidden_dim"],
        lr=config["lr"],
        t=config["t"],
        momentum=config["momentum"],
        patience=config["patience"],
        model=base,
        Dataset=BaselineDataset,
        Dataloader=dataloader_baseline,
        version='base',
        ruta_features_train=features["train"],
        ruta_features_val=features["val"],
        ruta_features_test1=features["test_cis"],
        ruta_features_test2=features["test_trans"],
        path_text_feat1=features["text_serengeti"],
        path_text_feat2=features["text_terra"],
        build_optimizer=build_optimizer,
        exp_name='exp_Base_Out_domain',
        subset_size=None
    )
    
    # Train
    try:
        results = model.train()
        
        print("\n" + "=" * 70)
        print("✅ TRAINING COMPLETED!")
        print("=" * 70)
        print(f"\nResults Summary:")
        print(f"  Best Model: {results['model_params_path']}")
        print(f"  Best Validation Accuracy: {results['best_val_acc']:.2f}%")
        print(f"  Final CIS Test Accuracy: {results['final_cis_test_acc']:.2f}%")
        print(f"\nConfusion matrix saved to: conf_matrix_cis_Base.csv")
        print("=" * 70 + "\n")
        
    except KeyboardInterrupt:
        print("\n\n⚠ Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
