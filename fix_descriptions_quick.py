#!/usr/bin/env python
"""
Quick fix: Create meaningful descriptions and extract BERT embeddings
This replaces the missing LLaVA descriptions with manually crafted class descriptions
"""

import torch
import json
from pathlib import Path
from transformers import BertModel, BertTokenizer
from tqdm import tqdm

print("\n" + "="*70)
print("  Creating Description Embeddings from Class Descriptions")
print("="*70)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\nDevice: {device}")

# Meaningful class descriptions
CLASS_DESCRIPTIONS = {
    'aardvark': "An aardvark is a nocturnal African mammal with a distinctive long snout, powerful claws for digging, and feeds on ants and termites using its sticky tongue.",
    'antelope': "An antelope is a swift, graceful hoofed mammal with slender powerful legs, curved horns, and streamlined body adapted for running.",
    'baboon': "A baboon is a large primate with a distinctive dog-like face, large canine teeth, thick muscular build, and lives in complex social groups.",
    'badger': "A badger is a stocky, powerful carnivore with short muscular legs, thick coarse fur, and distinctive white markings on its dark face.",
    'bat': "A bat is a unique flying mammal with wings made of stretched skin, echolocation abilities, and typically hangs upside down while resting.",
    'bear': "A bear is a large, heavy carnivore with thick fur covering its muscular body, a short tail, and powerful legs for climbing and running.",
    'bee-eater': "A bee-eater is a colorful bird with vibrant plumage, long pointed wings, and hunts insects in flight, particularly bees.",
    'bird': "A bird is a feathered biped with two wings, a beak, hollow bones, and is adapted for flight and perching on branches.",
    'boar': "A boar is a wild pig with a robust, muscular body, protruding tusks, dark coarse bristly fur, and strong rooting behavior.",
    'buffalo': "A buffalo is a large bovine mammal with a robust build, curved horns, dark brown coat, and lives in herds in African savannas."
}

class_names = list(CLASS_DESCRIPTIONS.keys())

# Load BERT
print("\n[CHECK] Loading BERT model...")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
model.to(device)
model.eval()

print("  [OK] BERT loaded")

# Extract embeddings for each class description
print("\n[CHECK] Extracting BERT embeddings for class descriptions...")
description_embeddings = {}

for class_name in class_names:
    description = CLASS_DESCRIPTIONS[class_name]
    
    # Tokenize
    tokens = tokenizer(description, return_tensors='pt', truncation=True, max_length=512)
    tokens = {k: v.to(device) for k, v in tokens.items()}
    
    # Extract embedding
    with torch.no_grad():
        output = model(**tokens)
        embedding = output.pooler_output.squeeze(0).cpu()
    
    description_embeddings[class_name] = embedding
    print(f"  {class_name}: {embedding.shape}")

# Now load feature files and update descriptions
print(f"\n[CHECK] Loading and updating feature files...")

features_dir = Path('features/Features_serengeti/standard_features')

for split in ['train', 'val', 'test']:
    feature_file = features_dir / f'Features_CATALOG_{split}_16.pt'
    if not feature_file.exists():
        print(f"  ⚠️  Skipping {split} - file not found")
        continue
    
    print(f"\n  Processing {split}...")
    
    # Load
    data = torch.load(feature_file, weights_only=False)
    print(f"    Loaded image features: {data['image_features'].shape}")
    print(f"    Old desc embeddings: {data['description_embeddings'].shape}")
    
    # Get target indices to map each sample to its class description
    targets = data['target_index']
    
    # Create new description embeddings based on class
    new_desc_embeddings = []
    for target_idx in targets:
        class_name = class_names[int(target_idx.item()) if isinstance(target_idx, torch.Tensor) else target_idx]
        new_desc_embeddings.append(description_embeddings[class_name])
    
    new_desc_embeddings = torch.stack(new_desc_embeddings)
    
    print(f"    New desc embeddings: {new_desc_embeddings.shape}")
    
    # Verify not all zeros
    sample = new_desc_embeddings[0]
    print(f"    Sample embedding sum: {sample.sum():.6f}")
    print(f"    Sample embedding mean abs: {sample.abs().mean():.6f}")
    print(f"    All zeros? {torch.all(sample == 0).item()}")
    
    # Update and save
    data['description_embeddings'] = new_desc_embeddings
    torch.save(data, feature_file)
    print(f"    [OK] Updated {feature_file.name}")

print("\n" + "="*70)
print("  COMPLETE! Description embeddings now contain real BERT features")
print("="*70)
print("\nNext steps:")
print("  1. Retrain the model with the updated features")
print("  2. Monitor if accuracy drops (expected) or improves (multimodal benefits)")
print("  3. Tune the weight_Clip parameter for optimal fusion")
print("\n")
