"""
CATALOG Modified - Optimized for 8 Classes (0, 3-9)
Removing problematic classes 1 & 2 (Bear, Badger)

Strategy:
- Filter training/val/test to only use 8 classes
- Classes kept: 0,3,4,5,6,7,8,9 (8 animal classes)
- Classes removed: 1,2 (Bear, Badger - were problematic)
- Expected result: Higher accuracy on remaining classes
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

def filter_classes(images, descriptions, labels, classes_to_keep):
    """Filter dataset to only include specified classes"""
    mask = torch.zeros(len(labels), dtype=torch.bool)
    for cls_id in classes_to_keep:
        mask |= (labels == cls_id)
    
    return images[mask], descriptions[mask], labels[mask]

def remap_labels(labels, classes_to_keep):
    """Remap original class IDs to new consecutive IDs (0 to num_classes-1)"""
    old_to_new = {old_id: new_id for new_id, old_id in enumerate(sorted(classes_to_keep))}
    remapped = torch.tensor([old_to_new[lbl.item()] for lbl in labels])
    return remapped

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Classes to keep (exclude 1 and 2)
    classes_to_keep = [0, 3, 4, 5, 6, 7, 8, 9]
    num_classes_filtered = len(classes_to_keep)
    
    # Configuration
    config = {
        'batch_size': 48,
        'learning_rate': 0.08,
        'momentum': 0.8,
        'num_epochs': 20,
        'feature_dim': 512,
    }
    
    print("\n" + "="*80)
    print("CATALOG MODIFIED - OPTIMIZED FOR 8 CLASSES")
    print("="*80)
    print(f"\nStrategy:")
    print(f"  Remove classes: [1 (Bear), 2 (Badger)]")
    print(f"  Keep classes: {classes_to_keep}")
    print(f"  Training classes: {num_classes_filtered}")
    print(f"\nConfig:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    
    # Load features
    print("\n[LOADING] Features...")
    train_dict = load_features('features/Features_serengeti/standard_features/Features_CATALOG_train_16.pt')
    val_dict = load_features('features/Features_serengeti/standard_features/Features_CATALOG_val_16.pt')
    test_dict = load_features('features/Features_serengeti/standard_features/Features_CATALOG_test_16.pt')
    text_features = load_features('features/Features_serengeti/standard_features/Text_features_16.pt')
    
    # Original data
    train_img_orig = train_dict['image_features'].float()
    train_desc_orig = train_dict['description_embeddings'].float()
    train_labels_orig = train_dict['target_index']
    
    val_img_orig = val_dict['image_features'].float()
    val_desc_orig = val_dict['description_embeddings'].float()
    val_labels_orig = val_dict['target_index']
    
    test_img_orig = test_dict['image_features'].float()
    test_desc_orig = test_dict['description_embeddings'].float()
    test_labels_orig = test_dict['target_index']
    
    # Filter to keep only selected classes
    print(f"\n[FILTERING] Keeping only classes {classes_to_keep}...")
    train_img, train_desc, train_labels = filter_classes(train_img_orig, train_desc_orig, train_labels_orig, classes_to_keep)
    val_img, val_desc, val_labels = filter_classes(val_img_orig, val_desc_orig, val_labels_orig, classes_to_keep)
    test_img, test_desc, test_labels = filter_classes(test_img_orig, test_desc_orig, test_labels_orig, classes_to_keep)
    
    # Remap labels to 0-7 range
    train_labels = remap_labels(train_labels, classes_to_keep)
    val_labels = remap_labels(val_labels, classes_to_keep)
    test_labels = remap_labels(test_labels, classes_to_keep)
    
    print(f"  Train: {train_img.shape[0]} samples (was {train_img_orig.shape[0]})")
    print(f"  Val:   {val_img.shape[0]} samples (was {val_img_orig.shape[0]})")
    print(f"  Test:  {test_img.shape[0]} samples (was {test_img_orig.shape[0]})")
    
    # Move to device
    train_img = train_img.to(device)
    train_desc = train_desc.to(device)
    train_labels = train_labels.to(device)
    
    val_img = val_img.to(device)
    val_desc = val_desc.to(device)
    val_labels = val_labels.to(device)
    
    test_img = test_img.to(device)
    test_desc = test_desc.to(device)
    test_labels = test_labels.to(device)
    
    # Text embeddings - filter to keep only selected classes
    txt_global_orig = text_features.float()  # [512, 10]
    txt_global_orig = txt_global_orig.t()  # [10, 512]
    
    txt_global = txt_global_orig[classes_to_keep]  # [8, 512] - keep only selected classes
    txt_global = txt_global.to(device)
    
    print(f"  Text embeddings: {txt_global.shape}")
    
    # Create model for 8 classes
    print("\n[MODEL] Creating CATALOG Modified (8 classes)...")
    model = CALOGModified(
        num_classes=num_classes_filtered,
        feature_dim=config['feature_dim'],
        desc_dim=768
    ).to(device).float()
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config['learning_rate'],
        momentum=config['momentum'],
        weight_decay=0.0
    )
    
    # Training
    print("\n[TRAINING] Starting training...")
    print("="*80)
    
    save_dir = os.path.join('Best/exp_CATALOG_Optimized_8Classes', 
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
    print("TEST EVALUATION (8 Classes Only)")
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
    
    print(f"\nTest Accuracy (8 classes): {test_acc:.2f}%")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}% (Epoch {best_epoch})")
    
    print(f"\nPer-class accuracy (Original class IDs):")
    class_names = {0: "Baboon", 3: "Bat", 4: "Bird", 5: "Buffalo", 6: "Boar", 7: "Bee-eater", 8: "Aardvark", 9: "Antelope"}
    for new_id in sorted(test_per_class.keys()):
        orig_id = classes_to_keep[new_id]
        acc = 100 * test_per_class[new_id]['correct'] / test_per_class[new_id]['total']
        name = class_names.get(orig_id, f"Class{orig_id}")
        print(f"  Class {orig_id} ({name:12s}): {acc:6.2f}%")
    
    print(f"\n✓ Training complete! Model saved to: {save_dir}")

if __name__ == '__main__':
    main()
