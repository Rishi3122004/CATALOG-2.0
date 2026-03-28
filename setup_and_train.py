"""
Complete setup and training pipeline for CATALOG Base Model
Guides through: feature extraction → training → evaluation
"""

import os
import sys
import subprocess
import torch
from pathlib import Path


def check_features_exist():
    """Check if all required features are extracted"""
    features_needed = [
        "features/Features_serengeti/standard_features/Features_CATALOG_train_16.pt",
        "features/Features_serengeti/standard_features/Features_CATALOG_val_16.pt",
        "features/Features_serengeti/standard_features/Features_CATALOG_test_16.pt",
        "features/Features_serengeti/standard_features/Text_features_16.pt",
        "features/Features_terra/standard_features/Features_CATALOG_cis_test_16.pt",
        "features/Features_terra/standard_features/Features_CATALOG_trans_test_16.pt",
        "features/Features_terra/standard_features/Text_features_16.pt",
    ]
    
    missing = []
    for feat in features_needed:
        if not os.path.exists(feat):
            missing.append(feat)
    
    if missing:
        return False, missing
    return True, []


def check_data_exists():
    """Check if dataset images exist"""
    data_paths = [
        "data/serengeti/img/Train",
        "data/serengeti/img/Val",
        "data/terra/img",
    ]
    
    for path in data_paths:
        if not os.path.exists(path):
            return False, path
    return True, None


def check_gpu():
    """Check GPU availability"""
    if torch.cuda.is_available():
        print(f"✓ GPU Available: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        return True
    else:
        print("⚠ GPU NOT available - training on CPU (will be slow)")
        return False


def main():
    print("=" * 70)
    print("  CATALOG Base Model - Complete Training Pipeline")
    print("=" * 70)
    
    # Step 1: Check prerequisites
    print("\n[1/3] Checking Prerequisites...")
    print("-" * 70)
    
    # Check GPU
    has_gpu = check_gpu()
    
    # Check data
    print("\n✓ Checking dataset...")
    data_ok, missing_data = check_data_exists()
    if not data_ok:
        print(f"❌ Dataset missing at: {missing_data}")
        sys.exit(1)
    print("✓ Dataset found")
    
    # Check features
    print("\n✓ Checking extracted features...")
    features_ok, missing_features = check_features_exist()
    
    if not features_ok:
        print(f"\n⚠ Missing {len(missing_features)} feature files:")
        for feat in missing_features[:3]:
            print(f"   - {feat}")
        if len(missing_features) > 3:
            print(f"   ... and {len(missing_features) - 3} more")
        
        print("\n📝 You need to extract features first.")
        print("\nRun this command to extract features:")
        print("   python extract_features_fixed.py")
        print("\nThen come back and run this script again.")
        sys.exit(1)
    
    print("✓ All features found!")
    
    # Step 2: Training configuration
    print("\n[2/3] Training Configuration")
    print("-" * 70)
    
    configs = {
        "1": {
            "name": "Quick Test (1 epoch, small batch)",
            "epochs": 1,
            "batch_size": 16,
            "patience": 1,
        },
        "2": {
            "name": "Standard Training (50 epochs)",
            "epochs": 50,
            "batch_size": 48,
            "patience": 5,
        },
        "3": {
            "name": "Full Training (100 epochs)",
            "epochs": 100,
            "batch_size": 48,
            "patience": 10,
        },
        "4": {
            "name": "Custom configuration",
            "custom": True,
        }
    }
    
    print("\nSelect training configuration:")
    for key, config in configs.items():
        print(f"  {key}. {config['name']}")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice not in configs:
        print("❌ Invalid choice")
        sys.exit(1)
    
    config = configs[choice]
    
    if config.get("custom"):
        print("\nEnter custom parameters:")
        epochs = int(input("  Epochs [50]: ") or "50")
        batch_size = int(input("  Batch size [48]: ") or "48")
        lr = float(input("  Learning rate [0.07641]: ") or "0.07641")
        patience = int(input("  Early stopping patience [5]: ") or "5")
    else:
        epochs = config["epochs"]
        batch_size = config["batch_size"]
        lr = 0.07641
        patience = config["patience"]
    
    print(f"\n✓ Configuration:")
    print(f"  - Epochs: {epochs}")
    print(f"  - Batch Size: {batch_size}")
    print(f"  - Learning Rate: {lr}")
    print(f"  - Early Stopping Patience: {patience}")
    
    # Step 3: Run training
    print("\n[3/3] Starting Training")
    print("-" * 70)
    
    try:
        cmd = [
            "python",
            "train_catalog_base.py",
            f"--epochs={epochs}",
            f"--batch_size={batch_size}",
            f"--lr={lr}",
            f"--patience={patience}",
            "--mode=train"
        ]
        
        print(f"\n🚀 Running: {' '.join(cmd)}\n")
        
        result = subprocess.run(cmd, check=True)
        
        print("\n" + "=" * 70)
        print("✅ TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print("\nResults saved in: Best/exp_Base_Out_domain/")
        print("\nNext steps:")
        print("  1. Review training results in Best/exp_Base_Out_domain/")
        print("  2. Evaluate on test set: python train_catalog_base.py --mode=test")
        print("  3. Check confusion matrix: conf_matrix_cis_Base.csv")
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Training failed with error code {e.returncode}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n⚠ Training interrupted by user")
        sys.exit(1)


if __name__ == "__main__":
    main()
