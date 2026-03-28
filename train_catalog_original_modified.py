"""
Training script for CATALOG Original Modified
Modifications:
1. Learnable alpha fusion weight (instead of fixed 0.6)
2. Learnable temperature scaling 
3. Layer normalization on embeddings (better gradient flow)
4. Increased dropout (help with problematic classes)
"""

import torch
import torch.nn as nn
import os
import time
from datetime import datetime
from models.CATALOG_Model_Modified import CALOGModified

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
    
    # Configuration
    config = {
        'batch_size': 48,
        'learning_rate': 0.08,
        'momentum': 0.8,
        'num_epochs': 20,
        'num_classes': 10,
        'feature_dim': 512,
    }
    
    print("\n" + "="*80)
    print("CATALOG MODIFIED - WITH LEARNABLE COMPONENTS")
    print("="*80)
    print(f"\nModifications:")
    print(f"  1. Learnable alpha fusion weight (was fixed 0.6)")
    print(f"  2. Learnable temperature scaling")
    print(f"  3. Layer normalization on embeddings")
    print(f"  4. Increased dropout (0.15)")
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
    train_desc = train_dict['description_embeddings'].to(device).float()
    train_labels = train_dict['target_index'].to(device)
    
    val_img = val_dict['image_features'].to(device).float()
    val_desc = val_dict['description_embeddings'].to(device).float()
    val_labels = val_dict['target_index'].to(device)
    
    test_img = test_dict['image_features'].to(device).float()
    test_desc = test_dict['description_embeddings'].to(device).float()
    test_labels = test_dict['target_index'].to(device)
    
    txt_global = text_features.to(device).float()  # [512, 10]
    txt_global = txt_global.t()  # [10, 512] - class embeddings
    
    print(f"  Train: {train_img.shape}, Val: {val_img.shape}, Test: {test_img.shape}")
    print(f"  Text embeddings: {txt_global.shape}")
    
    # Create model
    print("\n[MODEL] Creating CATALOG Modified...")
    model = CALOGModified(
        num_classes=config['num_classes'],
        feature_dim=config['feature_dim'],
        desc_dim=768  # BERT embeddings
    ).to(device).float()
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer (SGD for compatibility with paper)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config['learning_rate'],
        momentum=config['momentum'],
        weight_decay=0.0
    )
    
    # Training
    print("\n[TRAINING] Starting training...")
    print("="*80)
    
    save_dir = os.path.join('Best/exp_CATALOG_Original_Modified', 
                           f"training_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
    os.makedirs(save_dir, exist_ok=True)
    
    best_val_acc = 0.0
    best_epoch = 0
    best_model_path = None
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(config['num_epochs']):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0.0
        batch_count = 0
        
        for i in range(0, len(train_img), config['batch_size']):
            batch_img = train_img[i:i+config['batch_size']]
            batch_desc = train_desc[i:i+config['batch_size']]
            batch_labels = train_labels[i:i+config['batch_size']]
            
            optimizer.zero_grad()
            
            logits = model(batch_img, batch_desc, batch_labels, txt_global)
            loss = criterion(logits, batch_labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            _, predictions = logits.max(1)
            train_correct += predictions.eq(batch_labels).sum().item()
            batch_count += 1
        
        train_loss_avg = train_loss / batch_count
        train_acc = (train_correct / len(train_img)) * 100
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0.0
        batch_count = 0
        
        with torch.no_grad():
            for i in range(0, len(val_img), config['batch_size']):
                batch_img = val_img[i:i+config['batch_size']]
                batch_desc = val_desc[i:i+config['batch_size']]
                batch_labels = val_labels[i:i+config['batch_size']]
                
                logits = model(batch_img, batch_desc, batch_labels, txt_global)
                loss = criterion(logits, batch_labels)
                val_loss += loss.item()
                
                _, predictions = logits.max(1)
                val_correct += predictions.eq(batch_labels).sum().item()
                batch_count += 1
        
        val_loss_avg = val_loss / batch_count
        val_acc = (val_correct / len(val_img)) * 100
        
        # Print progress
        print(f"Epoch {epoch+1:2d} | Train: {train_acc:6.2f}% ({train_loss_avg:.4f}) | Val: {val_acc:6.2f}% ({val_loss_avg:.4f})", end="")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            best_model_path = os.path.join(save_dir, f'best_model_params_{epoch+1}_{int(val_acc*100)}.pth')
            torch.save(model.state_dict(), best_model_path)
            print(" ✓ SAVED")
        else:
            print()
    
    # Test evaluation
    print("\n" + "="*80)
    print("TEST EVALUATION")
    print("="*80)
    
    if best_model_path:
        model.load_state_dict(torch.load(best_model_path))
        print(f"Loaded best model from Epoch {best_epoch} (Val Acc: {best_val_acc:.2f}%)")
    
    model.eval()
    test_correct = 0.0
    test_per_class = {}
    
    with torch.no_grad():
        for i in range(0, len(test_img), config['batch_size']):
            batch_img = test_img[i:i+config['batch_size']]
            batch_desc = test_desc[i:i+config['batch_size']]
            batch_labels = test_labels[i:i+config['batch_size']]
            
            logits = model(batch_img, batch_desc, batch_labels, txt_global)
            _, predictions = logits.max(1)
            test_correct += predictions.eq(batch_labels).sum().item()
            
            # Per-class
            for pred, label in zip(predictions, batch_labels):
                label_int = label.item()
                if label_int not in test_per_class:
                    test_per_class[label_int] = {'correct': 0, 'total': 0}
                test_per_class[label_int]['total'] += 1
                if pred == label:
                    test_per_class[label_int]['correct'] += 1
    
    test_acc = (test_correct / len(test_img)) * 100
    
    print(f"\nTest Accuracy: {test_acc:.2f}%")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}% (Epoch {best_epoch})")
    
    print(f"\nPer-class accuracy:")
    for class_id in sorted(test_per_class.keys()):
        acc = 100 * test_per_class[class_id]['correct'] / test_per_class[class_id]['total']
        print(f"  Class {class_id}: {acc:6.2f}%")
    
    print(f"\n✓ Training complete! Model saved to: {save_dir}")

if __name__ == '__main__':
    main()

