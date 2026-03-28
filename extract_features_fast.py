"""
Fast batched feature extraction for wilddata
Processes images in batches for ~10x speedup
"""

import os
import torch
from transformers import BertModel, BertTokenizer
import clip
from PIL import Image
from pathlib import Path

#============= CONFIG =============
root_img = r'C:\Users\rishi\CATALOG\data\serengeti\img'
output_dir = r'C:\Users\rishi\CATALOG\features\Features_serengeti\standard_features'
os.makedirs(output_dir, exist_ok=True)

classes = ['elephant', 'gazellegrants', 'gazellethomsons', 'giraffe', 'guineafowl',
           'hartebeest', 'hyenaspotted', 'lionfemale', 'warthog', 'zebra']

BATCH_SIZE = 32

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\nUsing device: {device}")

# Load models
model_clip, preprocess_clip = clip.load('ViT-B/16', device)
tokenizer_bert = BertTokenizer.from_pretrained('bert-base-uncased')
model_bert = BertModel.from_pretrained('bert-base-uncased')
model_bert.to(device)
model_bert.eval()
model_clip.eval()

def extract_split(split_name):
    """Extract features for a train/val/test split using batching """
    print(f"\n{'='*70}")
    print(f"Extracting {split_name.upper()} features (batched)...")
    print('='*70)
    
    split_dir = os.path.join(root_img, split_name)
    
    image_features_list = []
    text_features_list = []
    targets_list = []
    
    total_processed = 0
    
    for class_idx, class_name in enumerate(classes):
        class_dir = os.path.join(split_dir, class_name)
        if not os.path.exists(class_dir):
            print(f"Warning: {class_name} directory not found")
            continue
        
        img_files = sorted([f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        print(f"\n{class_name}: {len(img_files)} images", end='', flush=True)
        
        # Process in batches
        for batch_start in range(0, len(img_files), BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, len(img_files))
            batch_files = img_files[batch_start:batch_end]
            
            # Load and preprocess images
            batch_images = []
            batch_targets = []
            batch_valid = []
            
            for i, img_file in enumerate(batch_files):
                try:
                    img_path = os.path.join(class_dir, img_file)
                    img = Image.open(img_path).convert('RGB')
                    img_tensor = preprocess_clip(img)
                    batch_images.append(img_tensor)
                    batch_targets.append(class_idx)
                    batch_valid.append(True)
                except Exception as e:
                    batch_valid.append(False)
            
            if not any(batch_valid):
                continue
            
            # Filter out failed loads
            batch_images = [batch_images[i] for i in range(len(batch_images)) if batch_valid[i]]
            batch_targets = [batch_targets[i] for i in range(len(batch_targets)) if batch_valid[i]]
            
            if not batch_images:
                continue
            
            batch_images = torch.stack(batch_images).to(device)
            
            # Extract CLIP features
            with torch.no_grad():
                image_feats = model_clip.encode_image(batch_images)
                image_feats = image_feats / image_feats.norm(dim=-1, keepdim=True)
                image_features_list.append(image_feats.cpu())
            
            # Extract BERT features for class name
            text_input = tokenizer_bert(class_name, return_tensors='pt').to(device)
            with torch.no_grad():
                text_output = model_bert(**text_input)
                text_feat = text_output.last_hidden_state.mean(dim=1)
                text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
            
            text_features_list.extend([text_feat.cpu()] * len(batch_images))
            targets_list.extend(batch_targets)
            total_processed += len(batch_images)
            
            if (batch_end) % 200 == 0 or batch_end == len(img_files):
                print(f".", end='', flush=True)
        
        print(f" ✓")
    
    # Save features
    if image_features_list:
        image_feats_final = torch.cat(image_features_list, dim=0)
        text_feats_final = torch.cat(text_features_list, dim=0)
        targets_final = torch.tensor(targets_list)
        
        feature_dict = {
            'image_features': image_feats_final,
            'description_embeddings': text_feats_final,
            'target_index': targets_final
        }
        
        out_file = os.path.join(output_dir, f'Features_CATALOG_{split_name.lower()}_16.pt')
        torch.save(feature_dict, out_file)
        
        size_mb = os.path.getsize(out_file) / (1024**2)
        print(f"\n✓ Saved {out_file} ({size_mb:.1f}MB, {len(targets_list)} samples)")
    
    return total_processed

# Extract splits
train_count = extract_split('Train')
val_count = extract_split('Val')
test_count = extract_split('Test')

# Class text features
print(f"\n{'='*70}")
print("Generating class text features...")
text_feats_all = []
for class_name in classes:
    text_input = tokenizer_bert(class_name, return_tensors='pt').to(device)
    with torch.no_grad():
        text_out = model_bert(**text_input)
        text_f = text_out.last_hidden_state.mean(dim=1)
        text_f = text_f / text_f.norm(dim=-1, keepdim=True)
    text_feats_all.append(text_f.cpu())

text_feats_final = torch.cat(text_feats_all, dim=0)
text_file = os.path.join(output_dir, 'Text_features_16.pt')
torch.save(text_feats_final, text_file)
print(f"✓ Saved {text_file}")

print(f"\n{'='*70}")
print(f"COMPLETE: Train={train_count}, Val={val_count}, Test={test_count}")
print('='*70)
