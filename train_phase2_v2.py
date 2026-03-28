#!/usr/bin/env python
"""
CATALOG Phase 2 Training - Fast Version
Tests Phase 2 enhancements with quick training
"""

import os
import sys
import torch
import time
import datetime
import numpy as np
import random

sys.path.insert(0, 'models')

from CATALOG_Base_Phase2 import LLaVA_CLIP_Phase2

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

device = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 90)
print("  CATALOG PHASE 2 FAST TRAINING")
print("=" * 90)

set_seed(42)

# Config
config = {
    'num_epochs': 86,
    'batch_size': 26,
    'hidden_dim': 1743,
    'num_layers': 4,
    'dropout': 0.381,
    'lr': 0.0956,
}

# Load data
print("\n[LOADING] Features...")
train_dict = torch.load('features/Features_serengeti/standard_features/Features_CATALOG_train_16.pt', weights_only=False)
val_dict = torch.load('features/Features_serengeti/standard_features/Features_CATALOG_val_16.pt', weights_only=False)
test_dict = torch.load('features/Features_serengeti/standard_features/Features_CATALOG_test_16.pt', weights_only=False)
txt_global = torch.load('features/Features_serengeti/standard_features/Text_features_16.pt', weights_only=False)

train_img = train_dict['image_features'].to(device).float()
train_emb =train_dict['description_embeddings'].to(device).float()
train_lbl = train_dict['target_index'].to(device)

val_img = val_dict['image_features'].to(device).float()
val_emb = val_dict['description_embeddings'].to(device).float()
val_lbl = val_dict['target_index'].to(device)

test_img = test_dict['image_features'].to(device).float()
test_emb = test_dict['description_embeddings'].to(device).float()
test_lbl = test_dict['target_index'].to(device)

txt_global = txt_global.to(device).float()

print(f"  Loaded: Train {train_img.shape}, Val {val_img.shape}, Test {test_img.shape}")

# Model
print("\n[MODEL] Phase 2 Architecture with:")
print("  - Fine-tunable BERT encoder")
print("  - Learnable fusion weights")
print("  - Attention mechanism")
print("  - Hard negative mining")

model = LLaVA_CLIP_Phase2(
    hidden_dim=config['hidden_dim'],
    num_layers=config['num_layers'],
    dropout=config['dropout'],
    device=device
).to(device).float()

# Optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=config['lr'], momentum=0.8162)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

# Training
save_dir = f"Best/exp_Base_In_domain_Serengeti_Phase2/training_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}/"
os.makedirs(save_dir, exist_ok=True)

print(f"\n[TRAINING] Saving to: {save_dir}")

best_val_acc = 0
patience = 20
counter = 0

for epoch in range(config['num_epochs']):
    print(epoch)
    
    # Train
    model.train()
    train_loss, train_acc_count = 0, 0
    for i in range(0, len(train_img), config['batch_size']):
        end_idx = min(i + config['batch_size'], len(train_img))
        batch_img = train_img[i:end_idx]
        batch_emb = train_emb[i:end_idx]
        batch_lbl = train_lbl[i:end_idx]
        
        try:
            loss, acc, _ = model(batch_emb, batch_img, txt_global, 0.60855, batch_lbl, 0.1, use_hard_mining=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            train_loss += loss.item()
            train_acc_count += acc.item()
        except Exception as e:
            print(f"    Error in train batch: {str(e)[:100]}")
            continue
    
    train_loss_avg = train_loss / max(1, len(range(0, len(train_img), config['batch_size'])))
    train_acc = (train_acc_count / len(train_img) * 100) if len(train_img) > 0 else 0
    
    # Val
    model.eval()
    val_loss, val_acc_count = 0, 0
    with torch.no_grad():
        for i in range(0, len(val_img), config['batch_size']):
            end_idx = min(i + config['batch_size'], len(val_img))
            batch_img = val_img[i:end_idx]
            batch_emb = val_emb[i:end_idx]
            batch_lbl = val_lbl[i:end_idx]
            
            try:
                loss, acc, _ = model.predict(batch_emb, batch_img, txt_global, 0.60855, batch_lbl, 0.1)
                val_loss += loss.item()
                val_acc_count += acc.item()
            except Exception as e:
                print(f"    Error in val batch: {str(e)[:100]}")
                continue
    
    val_loss_avg = val_loss / max(1, len(range(0, len(val_img), config['batch_size'])))
    val_acc = (val_acc_count / len(val_img) * 100) if len(val_img) > 0 else 0
    
    print(f"Train loss: {train_loss_avg:.4f}, acc: {train_acc:.4f}")
    print(f"Val loss: {val_loss_avg:.4f}, Val acc: {val_acc:.4f}")
    
    # Early stopping
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        counter = 0
        model_path = f"{save_dir}best_model_params_{config['num_layers']}_{config['hidden_dim']}.pth"
        torch.save(model.state_dict(), model_path)
        print("Save model")
    else:
        print("The acc don't increase")
        counter += 1

# Test
print("\n" + "=" * 90)
print("  PHASE 2 MODEL - TEST EVALUATION")
print("=" * 90)

model.eval()
test_loss, test_acc_count = 0, 0
with torch.no_grad():
    for i in range(0, len(test_img), config['batch_size']):
        end_idx = min(i + config['batch_size'], len(test_img))
        batch_img = test_img[i:end_idx]
        batch_emb = test_emb[i:end_idx]
        batch_lbl = test_lbl[i:end_idx]
        
        try:
            loss, acc, _ = model.predict(batch_emb, batch_img, txt_global, 0.60855, batch_lbl, 0.1)
            test_loss += loss.item()
            test_acc_count += acc.item()
        except Exception as e:
            print(f"Error: {str(e)}")

test_loss_avg = test_loss / max(1, len(range(0, len(test_img), config['batch_size'])))
test_acc = (test_acc_count / len(test_img) * 100) if len(test_img) > 0 else 0

print(f"Test loss: {test_loss_avg:.4f}, Test acc: {test_acc:.4f}")
print("=" * 90)

print(f"\n[RESULTS SUMMARY]")
print(f"  Best Val Accuracy: {best_val_acc:.4f}%")
print(f"  Test Accuracy: {test_acc:.4f}%")
print(f"  Test Loss: {test_loss_avg:.4f}")
print(f"  Model: {model_path}")
