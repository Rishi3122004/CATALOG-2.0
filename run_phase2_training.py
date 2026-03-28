#!/usr/bin/env python
"""
CATALOG Phase 2 Training - Executable Script
Trains enhanced model with all Phase 2 improvements
"""

import os
import sys
import torch
import time
import datetime
import numpy as np
import random

sys.path.insert(0, 'models')
sys.path.insert(0, 'train/Base')
sys.path.insert(0, 'data/serengeti')

from CATALOG_Base_Phase2 import LLaVA_CLIP_Phase2

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def load_features(path):
    """Load feature dictionary"""
    return torch.load(path, weights_only=False)

def train_phase2():
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("=" * 90)
    print("  CATALOG PHASE 2 TRAINING")
    print("  Enhanced Multimodal Learning with Advanced Fusion")
    print("=" * 90)
    
    print(f"\n[GPU] {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"[MEMORY] {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB" if torch.cuda.is_available() else "")
    
    # Configuration
    config = {
        'weight_Clip': 0.60855,
        'num_epochs': 86,
        'batch_size': 26,
        'num_layers': 4,
        'dropout': 0.381,
        'hidden_dim': 1743,
        'lr': 0.0956,
        't': 0.1,
        'momentum': 0.8162,
        'patience': 20,
        'hidden_dim_fusion': 128,
    }
    
    print("\n[CONFIG] Phase 2 Hyperparameters:")
    for key, val in config.items():
        print(f"  {key:.<30} {val}")
    
    # Load features
    print("\n[LOADING] Features...")
    train_dict = load_features('features/Features_serengeti/standard_features/Features_CATALOG_train_16.pt')
    val_dict = load_features('features/Features_serengeti/standard_features/Features_CATALOG_val_16.pt')
    test_dict = load_features('features/Features_serengeti/standard_features/Features_CATALOG_test_16.pt')
    text_features_global = load_features('features/Features_serengeti/standard_features/Text_features_16.pt')
    
    train_img = train_dict['image_features'].to(device).float()
    train_txt_emb = train_dict['description_embeddings'].to(device).float()
    train_labels = train_dict['target_index'].to(device)
    
    val_img = val_dict['image_features'].to(device).float()
    val_txt_emb = val_dict['description_embeddings'].to(device).float()
    val_labels = val_dict['target_index'].to(device)
    
    test_img = test_dict['image_features'].to(device).float()
    test_txt_emb = test_dict['description_embeddings'].to(device).float()
    test_labels = test_dict['target_index'].to(device)
    
    txt_global = text_features_global.to(device).float()
    
    print(f"  Train: {train_img.shape}, Val: {val_img.shape}, Test: {test_img.shape}")
    print(f"  Text embeddings: {txt_global.shape}")
    
    # Create Phase 2 model
    print("\n[MODEL] Creating Phase 2 Enhanced Architecture...")
    model = LLaVA_CLIP_Phase2(
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        dropout=config['dropout'],
        device=device,
        num_classes=10,
        enable_bert_tuning=True
    ).to(device).float()
    
    print("  Enhanced Features:")
    print("    [X] Fine-tunable BERT encoder")
    print("    [X] Learnable fusion weights (per-sample)")
    print("    [X] Attention-based fusion mechanism")
    print("    [X] Hard negative mining")
    print("    [X] Learnable temperature parameters")
    
    # Optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=config['lr'], momentum=config['momentum'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
    
    # Training
    unique_id = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_dir = f"Best/exp_Base_In_domain_Serengeti_Phase2/training_{unique_id}/"
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\n[TRAINING] Starting Phase 2 training...")
    print(f"  Save directory: {save_dir}")
    
    best_val_acc = 0
    counter = 0
    
    for epoch in range(config['num_epochs']):
        print(epoch)
        
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        batch_count = 0
        time_start = time.time()
        
        # Mini-batches
        for i in range(0, len(train_img), config['batch_size']):
            batch_img = train_img[i:i+config['batch_size']]
            batch_emb = train_txt_emb[i:i+config['batch_size']]
            batch_labels = train_labels[i:i+config['batch_size']]
            
            loss, acc, _ = model(batch_emb, batch_img, txt_global, config['weight_Clip'], 
                                batch_labels, config['t'], use_hard_mining=True)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Add gradient clipping
            optimizer.step()
            optimizer.zero_grad()
            
            train_loss += loss.item()
            train_correct += acc.item()
            batch_count += 1
        
        # Update learning rate ONCE per epoch (not per batch)
        scheduler.step()
        
        train_loss_avg = train_loss / batch_count
        train_acc = train_correct / len(train_img) * 100
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        batch_count = 0
        
        with torch.no_grad():
            for i in range(0, len(val_img), config['batch_size']):
                batch_img = val_img[i:i+config['batch_size']]
                batch_emb = val_txt_emb[i:i+config['batch_size']]
                batch_labels = val_labels[i:i+config['batch_size']]
                
                loss, acc, _ = model.predict(batch_emb, batch_img, txt_global, 
                                            config['weight_Clip'], batch_labels, config['t'])
                
                val_loss += loss.item()
                val_correct += acc.item()
                batch_count += 1
        
        val_loss_avg = val_loss / batch_count
        val_acc = val_correct / len(val_img) * 100
        
        time_epoch = time.time() - time_start
        
        print(f"Train loss: {train_loss_avg:.4f}, acc: {train_acc:.4f}")
        print(f"Val loss: {val_loss_avg:.4f}, Val acc: {val_acc:.4f}")
        print(f"Time for epoch [{time_epoch:.2f}s]")
        
        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            counter = 0
            model_path = os.path.join(save_dir, f"best_model_params_{config['num_layers']}_{config['hidden_dim']}.pth")
            torch.save(model.state_dict(), model_path)
            print("Save model")
        else:
            print("The acc don't increase")
            counter += 1
            if counter >= config['patience']:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
    
    # Test
    print("\n" + "=" * 90)
    print("  TEST EVALUATION - PHASE 2 MODEL")
    print("=" * 90)
    
    model.eval()
    test_loss = 0.0
    test_correct = 0
    batch_count = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for i in range(0, len(test_img), config['batch_size']):
            batch_img = test_img[i:i+config['batch_size']]
            batch_emb = test_txt_emb[i:i+config['batch_size']]
            batch_labels = test_labels[i:i+config['batch_size']]
            
            loss, acc, preds = model.predict(batch_emb, batch_img, txt_global, 
                                            config['weight_Clip'], batch_labels, config['t'])
            
            test_loss += loss.item()
            test_correct += acc.item()
            batch_count += 1
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(batch_labels.cpu().numpy())
    
    test_loss_avg = test_loss / batch_count
    test_acc = test_correct / len(test_img) * 100
    
    print(f"Test loss: {test_loss_avg:.4f}, Test acc: {test_acc:.4f}")
    print("=" * 90)
    
    return {
        'best_val': best_val_acc,
        'test_acc': test_acc,
        'test_loss': test_loss_avg,
        'model_path': model_path,
        'predictions': np.array(all_preds),
        'targets': np.array(all_targets)
    }

if __name__ == '__main__':
    results = train_phase2()
    print(f"\n[SUMMARY]")
    print(f"  Best Val Accuracy: {results['best_val']:.4f}%")
    print(f"  Test Accuracy: {results['test_acc']:.4f}%")
    print(f"  Model saved: {results['model_path']}")
