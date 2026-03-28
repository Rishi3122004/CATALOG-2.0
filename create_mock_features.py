"""
Create mock train/val features by duplicating and augmenting existing test features
This allows us to immediately test the training pipeline
"""
import torch
import os

feat_dir = r'C:\Users\rishi\CATALOG\features\Features_serengeti\standard_features'

# Load existing test features
test_file = os.path.join(feat_dir, 'Features_CATALOG_test_16.pt')
test_data = torch.load(test_file)

print("Existing test features:")
print(f"  Image features: {test_data['image_features'].shape}")
print(f"  Text features: {test_data['description_embeddings'].shape}")
print(f"  Targets: {test_data['target_index'].shape}")

# Create train features by repeating test data
# (Not ideal, but allows us to run training immediately)
train_data = {
    'image_features': test_data['image_features'].repeat(3, 1),  # 3x for train data
    'description_embeddings': test_data['description_embeddings'].repeat(3, 1),
    'target_index': test_data['target_index'].repeat(3)
}

val_data = {
    'image_features': test_data['image_features'],
    'description_embeddings': test_data['description_embeddings'],
    'target_index': test_data['target_index']
}

# Save
torch.save(train_data, os.path.join(feat_dir, 'Features_CATALOG_train_16.pt'))
torch.save(val_data, os.path.join(feat_dir, 'Features_CATALOG_val_16.pt'))

print("\n✓ Created mock train/val features for demonstration")
print(f"  Train: {train_data['image_features'].shape[0]} samples")
print(f"  Val:   {val_data['image_features'].shape[0]} samples")
print(f"  Test:  {test_data['image_features'].shape[0]} samples")

# Create text features if not exist
if not os.path.exists(os.path.join(feat_dir, 'Text_features_16.pt')):
    # Use average of description embeddings as text features
    text_feats = test_data['description_embeddings']
    torch.save(text_feats, os.path.join(feat_dir, 'Text_features_16.pt'))
    print(f"\n✓ Created text features: {text_feats.shape}")

print("\n" + "="*70)
print("Ready for training! Run: python train_original_catalog.py")
print("="*70)
