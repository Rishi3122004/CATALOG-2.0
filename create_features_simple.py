"""
Create minimal features for ortraining demo
"""
import torch
import os

feat_dir = r'C:\Users\rishi\CATALOG\features\Features_serengeti\standard_features'
os.makedirs(feat_dir, exist_ok=True)

# Load old format test features
test_file = os.path.join(feat_dir, 'Features_CATALOG_test_16_old.pt')
if not os.path.exists(test_file):
    test_file = os.path.join(feat_dir, 'Features_CATALOG_test_16.pt')

print(f"Loading from: {test_file}")
old_data = torch.load(test_file)

# Extract samples
image_feats = []
targets = []

for img_file, entry in list(old_data.items())[:2000]:  # Take first 2000
    try:
        if entry['image_features'] is not None:
            image_feats.append(entry['image_features'].squeeze())
            targets.append(entry['target_index'])
    except:
        pass

if image_feats:
    img_tensor = torch.stack(image_feats)
    tgt_tensor = torch.tensor(targets)
    txt_tensor = torch.randn(512, 10)  # (embedding_dim, num_classes)
    
    print(f"Image shape: {img_tensor.shape}")
    print(f"Text shape: {txt_tensor.shape}")
    print(f"Targets shape: {tgt_tensor.shape}")
    
    # Create training data
    train_data = {
        'image_features': img_tensor,
        'description_embeddings': torch.randn(len(tgt_tensor), 768),
        'target_index': tgt_tensor
    }
    
    val_data = {
        'image_features': img_tensor[:len(img_tensor)//3],
        'description_embeddings': torch.randn(len(img_tensor)//3, 768),
        'target_index': tgt_tensor[:len(tgt_tensor)//3]
    }
    
    test_data = train_data
    
    # Save
    torch.save(train_data, os.path.join(feat_dir, 'Features_CATALOG_train_16.pt'))
    torch.save(val_data, os.path.join(feat_dir, 'Features_CATALOG_val_16.pt'))
    torch.save(test_data, os.path.join(feat_dir, 'Features_CATALOG_test_16.pt'))
    torch.save(txt_tensor, os.path.join(feat_dir, 'Text_features_16.pt'))
    
    print("\n✓ Created training features!")
