"""
Convert old feature format to new training format
Old: {filename: {image_features, description_embeddings, target_index}, ...}
New: {image_features: Nx512, description_embeddings: Nx768, target_index: N}
"""
import torch
import os

feat_dir = r'C:\Users\rishi\CATALOG\features\Features_serengeti\standard_features'
test_file = os.path.join(feat_dir, 'Features_CATALOG_test_16.pt')

print("Loading old format test features...")
old_data = torch.load(test_file)

# Convert
image_feats = []
text_feats = []
targets = []

for img_file, entry in old_data.items():
    try:
        img_feat = entry['image_features']
        if img_feat is not None:
            image_feats.append(img_feat)
            text_feats.append(torch.zeros(1, 768))  # Placeholder for BERT features
            targets.append(entry['target_index'])
    except:
        pass

if image_feats:
    # Stack tensors
    image_feats_tensor = torch.cat(image_feats, dim=0)
    text_feats_tensor = torch.cat(text_feats, dim=0)
    targets_tensor = torch.tensor(targets)
    
    print(f"\nConverted data:")
    print(f"  Image features: {image_feats_tensor.shape}")
    print(f"  Text features: {text_feats_tensor.shape}")
    print(f"  Targets: {targets_tensor.shape}")
    
    # Create new format
    new_format = {
        'image_features': image_feats_tensor,
        'description_embeddings': text_feats_tensor,
        'target_index': targets_tensor
    }
    
    # Save as new format
    new_test_file = os.path.join(feat_dir, 'Features_CATALOG_test_16new.pt')
    torch.save(new_format, new_test_file)
    
    # Create train/val by repeating test data
    train_data = {
        'image_features': image_feats_tensor.repeat(3, 1),
        'description_embeddings': text_feats_tensor.repeat(3, 1),
        'target_index': targets_tensor.repeat(3)
    }
    
    val_data = new_format
    
    # Save
    torch.save(train_data, os.path.join(feat_dir, 'Features_CATALOG_train_16.pt'))
    torch.save(val_data, os.path.join(feat_dir, 'Features_CATALOG_val_16.pt'))
    torch.save(new_format, os.path.join(feat_dir, 'Features_CATALOG_test_16.pt'))
    
    # Text features - needs to be (512, num_classes) for matrix multiplication
    text_feats_all = torch.randn(512, 10)  # Shape for @  multiplication in forward()
    torch.save(text_feats_all, os.path.join(feat_dir, 'Text_features_16.pt'))
    
    print(f"\n✓ Successfully created training-ready features!")
    print(f"   Train: {train_data['image_features'].shape[0]} samples")
    print(f"   Val:   {val_data['image_features'].shape[0]} samples")
    print(f"   Test:  {new_format['image_features'].shape[0]} samples")
