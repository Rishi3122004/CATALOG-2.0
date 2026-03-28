#!/usr/bin/env python
"""
Retrain CATALOG model with REAL multimodal features (CLIP + BERT descriptions)
"""

import os
import sys
import torch
from pathlib import Path

# Set correct working directory
script_dir = Path(__file__).parent
os.chdir(script_dir)
sys.path.insert(0, str(script_dir))

from models import CATALOG_Projections as projections
from utils import BaselineDataset, dataloader_baseline, build_optimizer
from train.Base.Train_CATALOG_Projections_Serengeti import CATALOG_projections_serengeti


def main():
    print("\n" + "=" * 75)
    print("  CATALOG Base Model - Retraining with REAL Multimodal Features")
    print("  (CLIP Image Embeddings + BERT Description Embeddings)")
    print("=" * 75)
    
    # Check GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"\n[GPU Available] {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("\n[WARNING] GPU NOT available - using CPU")
    
    # Feature paths (all Serengeti, now with real description embeddings)
    features = {
        "train": "features/Features_serengeti/standard_features/Features_CATALOG_train_16.pt",
        "val": "features/Features_serengeti/standard_features/Features_CATALOG_val_16.pt",
        "test": "features/Features_serengeti/standard_features/Features_CATALOG_test_16.pt",
        "text": "features/Features_serengeti/standard_features/Text_features_16.pt",
    }
    
    # Check all features exist
    print("\n[CHECK] Verifying feature files...")
    all_exist = True
    for name, path in features.items():
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / (1024 ** 2)
            print(f"  [OK] {name:20} ({size_mb:7.1f} MB)")
        else:
            print(f"  [FAIL] {name:20} MISSING!")
            all_exist = False
    
    if not all_exist:
        print("\n[ERROR] Some feature files are missing!")
        return
    
    print("\n" + "-" * 75)
    print("Training Configuration:")
    print("-" * 75)
    config = {
        "weight_Clip": 0.60855,
        "num_epochs": 86,
        "batch_size": 26,
        "num_layers": 4,
        "dropout": 0.381,
        "hidden_dim": 1743,
        "lr": 0.0956,
        "t": 0.1,
        "momentum": 0.8162,
        "patience": 20,
    }
    for key, value in config.items():
        print(f"  {key:20} : {value}")
    
    print("\n[START] Beginning Multimodal Training...")
    print("=" * 75 + "\n")
    
    # Train
    trainer = CATALOG_projections_serengeti(
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
        model=projections,
        Dataset=BaselineDataset,
        Dataloader=dataloader_baseline,
        version='projection',  # Use a version that build_optimizer recognizes
        ruta_features_train=features["train"],
        ruta_features_val=features["val"],
        ruta_features_test=features["test"],
        path_text_feat=features["text"],
        build_optimizer=build_optimizer,
        exp_name='exp_Base_In_domain_Serengeti_Multimodal'
    )
    
    trainer.train()


if __name__ == '__main__':
    main()