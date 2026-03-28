"""
Training script for original CATALOG model from paper
"""

import torch
import torch.nn as nn
import os
import time
from datetime import datetime
from models.CATALOG_Original_Paper import CATALOG_Original

def load_features(feature_path):
    """Load pre-computed features."""
    if os.path.exists(feature_path):
        return torch.load(feature_path, weights_only=False)
    else:
        raise FileNotFoundError(f"Feature file not found: {feature_path}")

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Configuration (from paper + extended training)
    config = {
        'alpha': 0.6,              # Fusion weight (paper: 0.6)
        'temperature': 0.1,        # Temperature parameter
        'batch_size': 48,          # Paper: batch size 48
        'learning_rate': 0.08,     # Paper: learning rate 0.08
        'momentum': 0.8,           # Paper: momentum 0.8
        'num_epochs': 20,          # Extended: was 8 in paper, now 20 for better training
        'num_classes': 10,
        'feature_dim': 512,
    }
    
    print("\n" + "="*80)
    print("CATALOG ORIGINAL TRAINING - FROM PAPER")
    print("="*80)
    print(f"\nConfig:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    
    # Load features
    print("\n[LOADING] Features...")
    train_dict = load_features('features/Features_serengeti/standard_features/Features_CATALOG_train_16.pt')
    val_dict = load_features('features/Features_serengeti/standard_features/Features_CATALOG_val_16.pt')
    test_dict = load_features('features/Features_serengeti/standard_features/Features_CATALOG_test_16.pt')
    text_features = load_features('features/Features_serengeti/standard_features/Text_features_16.pt')
    
    # Prepare data
    train_img = train_dict['image_features'].to(device).float()
    train_emb = train_dict['description_embeddings'].to(device).float()
    train_labels = train_dict['target_index'].to(device)
    
    val_img = val_dict['image_features'].to(device).float()
    val_emb = val_dict['description_embeddings'].to(device).float()
    val_labels = val_dict['target_index'].to(device)
    
    test_img = test_dict['image_features'].to(device).float()
    test_emb = test_dict['description_embeddings'].to(device).float()
    test_labels = test_dict['target_index'].to(device)
    
    txt_global = text_features.to(device).float()  # [512, 10]
    # Transpose to [10, 512] for class embeddings
    txt_global = txt_global.t()  # Now [10, 512]
    
    print(f"  Train: {train_img.shape}, Val: {val_img.shape}, Test: {test_img.shape}")
    print(f"  Text embeddings: {txt_global.shape}")
    
    # Create model
    print("\n[MODEL] Creating CATALOG (Original from Paper)...")
    model = CATALOG_Original(
        feature_dim=config['feature_dim'],
        num_classes=config['num_classes'],
        alpha=config['alpha'],
        temperature=config['temperature']
    ).to(device).float()
    
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer (SGD as in paper)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config['learning_rate'],
        momentum=config['momentum'],
        weight_decay=0.0
    )
    
    # Training
    print("\n[TRAINING] Starting CATALOG training...")
    print("="*80)
    
    save_dir = os.path.join('Best/exp_CATALOG_Original', 
                           f"training_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
    os.makedirs(save_dir, exist_ok=True)
    
    best_val_acc = 0.0
    best_epoch = 0
    
    for epoch in range(config['num_epochs']):
        epoch_start = time.time()
        
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0.0
        batch_count = 0
        
        for i in range(0, len(train_img), config['batch_size']):
            batch_img = train_img[i:i+config['batch_size']]
            batch_emb = train_emb[i:i+config['batch_size']]
            batch_labels = train_labels[i:i+config['batch_size']]
            
            optimizer.zero_grad()
            loss, acc, _ = model(txt_global, batch_img, batch_emb, batch_labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            train_correct += acc
            batch_count += 1
        
        train_loss_avg = train_loss / batch_count
        train_acc = train_correct / batch_count * 100
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0.0
        batch_count = 0
        
        with torch.no_grad():
            for i in range(0, len(val_img), config['batch_size']):
                batch_img = val_img[i:i+config['batch_size']]
                batch_emb = val_emb[i:i+config['batch_size']]
                batch_labels = val_labels[i:i+config['batch_size']]
                
                loss, acc, _ = model(txt_global, batch_img, batch_emb, batch_labels)
                val_loss += loss.item()
                val_correct += acc
                batch_count += 1
        
        val_loss_avg = val_loss / batch_count
        val_acc = val_correct / batch_count * 100
        
        epoch_time = time.time() - epoch_start
        
        print(f"Epoch {epoch+1}/{config['num_epochs']} | "
              f"Train Loss: {train_loss_avg:.4f}, Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss_avg:.4f}, Acc: {val_acc:.2f}% | "
              f"Time: {epoch_time:.2f}s")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            model_path = os.path.join(save_dir, 'best_model_catalog_original.pth')
            torch.save(model.state_dict(), model_path)
            print(f"  ✓ Saved best model (Val Acc: {val_acc:.2f}%)")
    
    print("\n" + "="*80)
    print(f"Training complete! Best validation accuracy: {best_val_acc:.2f}% (Epoch {best_epoch})")
    print("="*80)
    
    # Test evaluation
    print("\n[TEST] Evaluating on test set...")
    
    # Load best model
    best_model_path = os.path.join(save_dir, 'best_model_catalog_original.pth')
    model.load_state_dict(torch.load(best_model_path, weights_only=False))
    model.eval()
    
    test_correct = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for i in range(0, len(test_img), config['batch_size']):
            batch_img = test_img[i:i+config['batch_size']]
            batch_emb = test_emb[i:i+config['batch_size']]
            batch_labels = test_labels[i:i+config['batch_size']]
            
            _, _, preds = model(txt_global, batch_img, batch_emb, batch_labels)
            test_correct += (preds == batch_labels).float().sum().item()
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(batch_labels.cpu().numpy())
    
    test_acc = test_correct / len(test_img) * 100
    
    # Print results
    print("\n" + "="*80)
    print("CATALOG ORIGINAL - FINAL RESULTS")
    print("="*80)
    print(f"\nTest Accuracy: {test_acc:.2f}%")
    
    # Per-class accuracy
    print("\nPer-class accuracy:")
    import numpy as np
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    for cls in range(config['num_classes']):
        cls_mask = all_targets == cls
        if cls_mask.sum() > 0:
            cls_acc = (all_preds[cls_mask] == all_targets[cls_mask]).mean() * 100
            print(f"  Class {cls}: {cls_acc:.2f}%")
    
    print(f"\nModel saved at: {save_dir}")
    print("="*80)
    
    return test_acc, best_val_acc

if __name__ == '__main__':
    test_acc, val_acc = main()
