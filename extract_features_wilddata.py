"""
Direct feature extraction for wilddata with 10 classes
Extracts CLIP + BERT embeddings for all images in Train/Val/Test splits
"""

import os
import torch
from transformers import BertModel, BertTokenizer
import clip
from PIL import Image
import json
from pathlib import Path

# Configuration
root_img = r'C:\Users\rishi\CATALOG\data\serengeti\img'
output_dir = r'C:\Users\rishi\CATALOG\features\Features_serengeti\standard_features'
os.makedirs(output_dir, exist_ok=True)

class_indices = ['elephant', 'gazellegrants', 'gazellethomsons', 'giraffe', 'guineafowl',
                 'hartebeest', 'hyenaspotted', 'lionfemale', 'warthog', 'zebra']

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load models
print("Loading CLIP...")
model_clip, preprocess_clip = clip.load('ViT-B/16', device)

print("Loading BERT...")
tokenizer_bert = BertTokenizer.from_pretrained('bert-base-uncased')
model_bert = BertModel.from_pretrained('bert-base-uncased')
model_bert.to(device)

def extract_split(split_name):
    """Extract features for a train/val/test split """
    print(f"\n{'='*60}")
    print(f"Extracting {split_name.upper()} features...")
    print('='*60)
    
    split_dir = os.path.join(root_img, split_name)
    
    all_image_feats = []
    all_text_feats = []
    all_targets = []
    
    total_processed = 0
    
    for class_idx, class_name in enumerate(class_indices):
        class_dir = os.path.join(split_dir, class_name)
        if not os.path.exists(class_dir):
            print(f"Warning: {class_name} folder not found")
            continue
        
        img_files = sorted([f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        print(f"\n{class_name}: {len(img_files)} images")
        
        for img_idx, img_file in enumerate(img_files):
            img_path = os.path.join(class_dir, img_file)
            
            try:
                # Extract CLIP features
                image = Image.open(img_path).convert('RGB')
                image_input = preprocess_clip(image).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    image_features = model_clip.encode_image(image_input)
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
                # Extract BERT features from class name
                text_input = tokenizer_bert(class_name, return_tensors='pt').to(device)
                with torch.no_grad():
                    text_output = model_bert(**text_input)
                    text_features = text_output.last_hidden_state.mean(dim=1)
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                
                all_image_feats.append(image_features.cpu())
                all_text_feats.append(text_features.cpu())
                all_targets.append(class_idx)
                
                total_processed += 1
                if (img_idx + 1) % 100 == 0:
                    print(f"  Processed {img_idx + 1}/{len(img_files)}")
            
            except Exception as e:
                print(f"  Error processing {img_file}: {e}")
                continue
        
        print(f"  ✓ Completed {class_name}")
    
    # Save features
    if all_image_feats:
        image_features_tensor = torch.cat(all_image_feats, dim=0)
        text_features_tensor = torch.cat(all_text_feats, dim=0)
        targets_tensor = torch.tensor(all_targets)
        
        feature_dict = {
            'image_features': image_features_tensor,
            'description_embeddings': text_features_tensor,
            'target_index': targets_tensor
        }
        
        output_file = os.path.join(output_dir, f'Features_CATALOG_{split_name.lower()}_16.pt')
        torch.save(feature_dict, output_file)
        print(f"\n✓ Saved: {output_file} ({len(all_targets)} samples)")
    
    return total_processed

# Extract all splits
train_count = extract_split('Train')
val_count = extract_split('Val')
test_count = extract_split('Test')

print(f"\n{'='*60}")
print(f"FEATURE EXTRACTION COMPLETE")
print(f"{'='*60}")
print(f"Train: {train_count} samples")
print(f"Val:   {val_count} samples")
print(f"Test:  {test_count} samples")
print(f"Total: {train_count + val_count + test_count} samples")

# Also save combined text features (all class embeddings)
print(f"\nGenerating text features for all {len(class_indices)} classes...")
text_feats_all = []
for class_name in class_indices:
    text_input = tokenizer_bert(class_name, return_tensors='pt').to(device)
    with torch.no_grad():
        text_output = model_bert(**text_input)
        text_feat = text_output.last_hidden_state.mean(dim=1)
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
    text_feats_all.append(text_feat.cpu())

text_features_all = torch.cat(text_feats_all, dim=0)
text_feat_file = os.path.join(output_dir, 'Text_features_16.pt')
torch.save(text_features_all, text_feat_file)
print(f"✓ Saved: {text_feat_file} ({text_features_all.shape})")

print(f"\n{'='*60}")
print("All features extracted successfully!")
print('='*60)
