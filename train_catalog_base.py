"""
Training script for CATALOG Base Model (Original)
Trains on Serengeti dataset with out-of-domain testing on Terra Incognita
"""

import os
import sys
import torch
from pathlib import Path

# Import model and training components
from models import CATALOG_Base as base
from utils import BaselineDataset, dataloader_baseline, build_optimizer
from train.Base.Train_CATALOG_Base_out_domain import CATALOG_base


def train_catalog_base(
    dataset_name="serengeti",
    weights_clip=0.6,
    num_epochs=50,
    batch_size=48,
    num_layers=1,
    hidden_dim=1045,
    dropout=0.27822,
    learning_rate=0.07641,
    temperature=0.1,
    momentum=0.8409,
    patience=5,
    subset_size=None,
    mode="train"
):
    """
    Train the original CATALOG Base model
    
    Args:
        dataset_name: 'serengeti' or 'terra' (serengeti for original training)
        weights_clip: Weight for CLIP loss
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        num_layers: Number of MLP layers
        hidden_dim: Hidden dimension of MLP
        dropout: Dropout rate
        learning_rate: Learning rate
        temperature: Temperature for contrastive loss
        momentum: SGD momentum
        patience: Early stopping patience
        subset_size: Limit dataset size (None for full dataset)
        mode: 'train', 'test', or 'test_top3'
    """
    
    print(f"🚀 Starting CATALOG Base Model Training")
    print(f"   Dataset: {dataset_name}")
    print(f"   Epochs: {num_epochs}, Batch Size: {batch_size}")
    print(f"   Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print("-" * 60)
    
    # Feature paths for Serengeti (original training)
    ruta_features_train = "features/Features_serengeti/standard_features/Features_CATALOG_train_16.pt"
    ruta_features_val = "features/Features_serengeti/standard_features/Features_CATALOG_val_16.pt"
    ruta_features_test1 = "features/Features_terra/standard_features/Features_CATALOG_cis_test_16.pt"
    ruta_features_test2 = "features/Features_terra/standard_features/Features_CATALOG_trans_test_16.pt"
    path_text_feat1 = "features/Features_serengeti/standard_features/Text_features_16.pt"
    path_text_feat2 = "features/Features_terra/standard_features/Text_features_16.pt"
    
    # Verify feature files exist
    feature_files = [
        ruta_features_train, ruta_features_val, ruta_features_test1,
        ruta_features_test2, path_text_feat1, path_text_feat2
    ]
    
    print("\n✓ Checking feature files...")
    for feat_file in feature_files:
        if not os.path.exists(feat_file):
            print(f"❌ Missing: {feat_file}")
            print("\nPlease extract features first using extract_features_fixed.py")
            sys.exit(1)
        print(f"  ✓ {Path(feat_file).name}")
    
    print("\n✓ All feature files found!")
    print("-" * 60)
    
    # Initialize training model
    model = CATALOG_base(
        weight_Clip=weights_clip,
        num_epochs=num_epochs,
        batch_size=batch_size,
        num_layers=num_layers,
        dropout=dropout,
        hidden_dim=hidden_dim,
        lr=learning_rate,
        t=temperature,
        momentum=momentum,
        patience=patience,
        model=base,
        Dataset=BaselineDataset,
        Dataloader=dataloader_baseline,
        version='base',
        ruta_features_train=ruta_features_train,
        ruta_features_val=ruta_features_val,
        ruta_features_test1=ruta_features_test1,
        ruta_features_test2=ruta_features_test2,
        path_text_feat1=path_text_feat1,
        path_text_feat2=path_text_feat2,
        build_optimizer=build_optimizer,
        exp_name=f'exp_Base_Out_domain',
        subset_size=subset_size
    )
    
    # Run training
    if mode == "train":
        print("\n📚 Starting training...")
        results = model.train()
        print("\n" + "=" * 60)
        print("✅ Training Complete!")
        print(f"   Best Model: {results['model_params_path']}")
        print(f"   Best Val Acc: {results['best_val_acc']:.2f}%")
        print(f"   Final CIS Test Acc: {results['final_cis_test_acc']:.2f}%")
        print("=" * 60)
        return results
    
    elif mode == "test" and os.path.exists("models/CATALOG_Base.pth"):
        print("\n🧪 Running test evaluation...")
        model.prueba_model("models/CATALOG_Base.pth")
    
    elif mode == "test_top3" and os.path.exists("models/CATALOG_Base.pth"):
        print("\n🎯 Running top-3 accuracy evaluation...")
        model.prueba_model_top_3("models/CATALOG_Base.pth")
    
    else:
        print(f"❌ Invalid mode: {mode}")
        sys.exit(1)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train CATALOG Base Model')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=48, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.07641, help='Learning rate')
    parser.add_argument('--hidden_dim', type=int, default=1045, help='Hidden dimension')
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience')
    parser.add_argument('--mode', type=str, default='train', help='train/test/test_top3')
    parser.add_argument('--subset_size', type=int, default=None, help='Limit dataset size (for debugging)')
    
    args = parser.parse_args()
    
    train_catalog_base(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        hidden_dim=args.hidden_dim,
        patience=args.patience,
        mode=args.mode,
        subset_size=args.subset_size
    )
