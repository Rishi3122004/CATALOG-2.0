"""
Train original CATALOG model (no fusion head) on new balanced dataset
"""

import os
import sys

# Add paths
sys.path.insert(0, r'C:\Users\rishi\CATALOG')

import torch
from models import CATALOG_Base as md
from train.Base.Train_CATALOG_Base_out_domain import CATALOG_base
import utils

def simple_dataloader(feature_file, batch_size, dataset_class, subset_size=None):
    """Load features and create dataloader"""
    data = torch.load(feature_file)
    dataset = dataset_class(data)
    
    # If subset size specified, take random subset
    if subset_size and subset_size < len(dataset):
        indices = torch.randperm(len(dataset))[:subset_size]
        dataset = torch.utils.data.Subset(dataset, indices)
    
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

class SimpleDataset(torch.utils.data.Dataset):
    """Simple dataset wrapper for loaded features"""
    def __init__(self, feature_dict):
        self.image_features = feature_dict['image_features']
        self.description_embeddings = feature_dict['description_embeddings']
        self.target_index = feature_dict['target_index']
        self.num_samples = len(self.target_index)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return (
            self.image_features[idx],
            self.description_embeddings[idx],
            self.target_index[idx]
        )

# Paths
features_base = r'C:\Users\rishi\CATALOG\features\Features_serengeti\standard_features'
ruta_features_train = os.path.join(features_base, 'Features_CATALOG_train_16.pt')
ruta_features_val = os.path.join(features_base, 'Features_CATALOG_val_16.pt')
ruta_features_test = os.path.join(features_base, 'Features_CATALOG_test_16.pt')
path_text_feat = os.path.join(features_base, 'Text_features_16.pt')

# Hyperparameters
weight_Clip = 0.5
num_epochs = 20
batch_size = 48
num_layers = 1
dropout = 0.27822
hidden_dim = 1045
lr = 0.03
temperature = 0.2
momentum = 0.9
patience = 7

print("\n" + "="*70)
print("TRAINING ORIGINAL CATALOG MODEL (WITHOUT FUSION HEAD)")
print("="*70)
print(f"\nConfiguration:")
print(f"  weight_Clip: {weight_Clip} (50% CLIP, 50% BERT)")
print(f"  lr: {lr}")
print(f"  enable_classifier_fusion: False")

# Model kwargs - disable fusion for original model
model_kwargs = {'enable_classifier_fusion': False}

# Initialize and train
trainer = CATALOG_base(
    weight_Clip=weight_Clip,
    num_epochs=num_epochs,
    batch_size=batch_size,
    num_layers=num_layers,
    dropout=dropout,
    hidden_dim=hidden_dim,
    lr=lr,
    t=temperature,
    momentum=momentum,
    patience=patience,
    model=md,
    Dataset=SimpleDataset,
    Dataloader=simple_dataloader,
    version='base',  # Must be lowercase!
    ruta_features_train=ruta_features_train,
    ruta_features_val=ruta_features_val,
    ruta_features_test1=ruta_features_test,
    ruta_features_test2=ruta_features_test,
    path_text_feat1=path_text_feat,
    path_text_feat2=path_text_feat,
    build_optimizer=utils.build_optimizer,
    exp_name='Original_CATALOG_Wilddata',
    model_kwargs=model_kwargs
)

print("\nStarting training...")
metrics = trainer.train()

print("\n" + "="*70)
print("RESULTS")
print("="*70)
if metrics:
    print(f"\n✓ Best Validation Accuracy: {metrics.get('best_val_acc', 'N/A'):.4f}")
    print(f"✓ Final Test Accuracy: {metrics.get('final_cis_test_acc', 'N/A'):.4f}")
    print(f"✓ Model Checkpoint: {metrics.get('model_params_path', 'N/A')}")
