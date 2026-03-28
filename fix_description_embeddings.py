#!/usr/bin/env python
"""
Re-extract description embeddings from LLaVA descriptions JSON files
"""

import os
import torch
import json
from pathlib import Path
from transformers import BertModel, BertTokenizer
from tqdm import tqdm

print("\n" + "="*70)
print("  Extracting REAL Description Embeddings from LLaVA JSON")
print("="*70)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\nDevice: {device}")

# Load BERT
print("\n[CHECK] Loading BERT model...")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
model.to(device)
model.eval()

# Class names mapping
CLASS_NAMES = [
    'aardvark', 'antelope', 'baboon', 'badger', 'bat',
    'bear', 'bee-eater', 'bird', 'boar', 'buffalo'
]

def extract_description_embeddings(split_name):
    """Extract description embeddings for a split"""
    print(f"\n[CHECK] Extracting {split_name.upper()} descriptions...")
    
    desc_embeddings_list = []
    targets_list = []
    missing_count = 0
    total_count = 0
    
    descriptions_dir = Path(f'data/serengeti/descriptions/{split_name}')
    
    if not descriptions_dir.exists():
        print(f"  ⚠️  Descriptions dir not found: {descriptions_dir}")
        return None, None
    
    # Iterate through all class directories
    for class_idx, class_name in enumerate(CLASS_NAMES):
        class_desc_dir = descriptions_dir / class_name
        if not class_desc_dir.exists():
            print(f"  ⚠️  Class dir not found: {class_desc_dir}")
            continue
        
        json_files = list(class_desc_dir.glob('*.json'))
        print(f"  {class_name}: {len(json_files)} descriptions")
        
        for json_file in tqdm(json_files, desc=f"  {class_name}", leave=False):
            total_count += 1
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                if 'description' not in data:
                    missing_count += 1
                    desc_embeddings_list.append(torch.zeros(768, device=device))
                    targets_list.append(class_idx)
                    continue
                
                description = data['description']
                
                # Tokenize and extract embedding
                tokens = tokenizer(description, return_tensors='pt', truncation=True, max_length=512)
                tokens = {k: v.to(device) for k, v in tokens.items()}
                
                with torch.no_grad():
                    output = model(**tokens)
                    embedding = output.pooler_output.squeeze(0).cpu()
                
                desc_embeddings_list.append(embedding)
                targets_list.append(class_idx)
                
            except Exception as e:
                missing_count += 1
                desc_embeddings_list.append(torch.zeros(768, device=device))
                targets_list.append(class_idx)
                print(f"    Error processing {json_file}: {e}")
    
    if not desc_embeddings_list:
        return None, None
    
    desc_embeddings = torch.stack(desc_embeddings_list)
    targets = torch.tensor(targets_list)
    
    print(f"    Extracted: {len(desc_embeddings_list)} samples ({missing_count} placeholders)")
    print(f"    Shape: {desc_embeddings.shape}")
    
    return desc_embeddings, targets

# Extract for all splits
all_splits = {}
for split in ['train', 'val', 'test']:
    embeddings, targets = extract_description_embeddings(split)
    if embeddings is not None:
        all_splits[split] = {'embeddings': embeddings, 'targets': targets}

if all_splits:
    # Load existing feature files and UPDATE description embeddings
    print(f"\n[CHECK] Updating feature files with real description embeddings...")
    
    features_dir = Path('features/Features_serengeti/standard_features')
    
    for split in ['train', 'val', 'test']:
        if split not in all_splits:
            continue
        
        feature_file = features_dir / f'Features_CATALOG_{split}_16.pt'
        if not feature_file.exists():
            print(f"  ⚠️  Feature file not found: {feature_file}")
            continue
        
        # Load existing features
        data = torch.load(feature_file, weights_only=False)
        print(f"\n  Loading {split}:")
        print(f"    Image features: {data['image_features'].shape}")
        print(f"    Old desc features: {data['description_embeddings'].shape}")
        
        # Replace with real descriptions
        new_desc_emb = all_splits[split]['embeddings']
        
        # Handle size mismatch (in case descriptions weren't extracted for all samples)
        if len(new_desc_emb) < len(data['image_features']):
            print(f"    ⚠️  Description count mismatch: {len(new_desc_emb)} vs {len(data['image_features'])}")
            # Pad with repeated descriptions if needed
            while len(new_desc_emb) < len(data['image_features']):
                indices = torch.randperm(len(new_desc_emb))[:len(data['image_features'])-len(new_desc_emb)]
                new_desc_emb = torch.cat([new_desc_emb, new_desc_emb[indices]], dim=0)
        
        # Update and save
        data['description_embeddings'] = new_desc_emb[:len(data['image_features'])]
        
        torch.save(data, feature_file)
        print(f"    [OK] Updated {split}: desc features now {data['description_embeddings'].shape}")
        
        # Verify
        updated_data = torch.load(feature_file, weights_only=False)
        sample_desc = updated_data['description_embeddings'][0]
        is_all_zeros = torch.all(sample_desc == 0).item()
        print(f"    [VERIFY] Sample desc all zeros? {is_all_zeros}")
        print(f"    [VERIFY] Sample desc sum: {sample_desc.sum():.6f}")

print("\n" + "="*70)
print("  Update Complete!")
print("="*70 + "\n")
