"""
Fixed feature extraction script for new 10-class balanced dataset
Extracts CLIP image features and BERT description features
"""

import os
import torch
from transformers import BertModel, BertTokenizer
import clip
import json
from PIL import Image
from pathlib import Path

# Set up absolute paths
SCRIPT_DIR = Path(__file__).resolve().parent
CATALOG_ROOT = SCRIPT_DIR.parent.parent
DATA_ROOT = CATALOG_ROOT / "data" / "serengeti" / "img"
FEATURES_DIR = CATALOG_ROOT / "features" / "Features_serengeti" / "standard_features"

# Ensure output directory exists
FEATURES_DIR.mkdir(parents=True, exist_ok=True)

# 10 new classes
CLASS_INDICES = [
    'elephant', 'gazellegrants', 'gazellethomsons', 'giraffe', 'guineafowl',
    'hartebeest', 'hyenaspotted', 'lionfemale', 'warthog', 'zebra'
]

# Text descriptions for each class
CLASS_DESCRIPTIONS = {
    'elephant': 'A large gray mammal with a long trunk and big ears',
    'gazellegrants': 'A medium-sized antelope with curved horns and tan fur',
    'gazellethomsons': 'A small gazelle with distinctive black stripes',
    'giraffe': 'A tall mammal with a long neck and distinctive spotted pattern',
    'guineafowl': 'A dark bird with white spots and a distinctive head',
    'hartebeest': 'A large antelope with a horse-like head and sloped back',
    'hyenaspotted': 'A large carnivore with powerful jaws and spotted fur',
    'lionfemale': 'A large golden female cat, lioness without a mane',
    'warthog': 'A boar with prominent tusks and coarse reddish-brown fur',
    'zebra': 'A striped equid with black and white horizontal stripes'
}

def extract_features():
    """Extract CLIP image features and BERT text features"""
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load models
    print("Loading CLIP model...")
    model_clip, preprocess_clip = clip.load("ViT-B/16", device)
    model_clip.to(device)
    
    print("Loading BERT model...")
    tokenizer_bert = BertTokenizer.from_pretrained('bert-base-uncased')
    model_bert = BertModel.from_pretrained('bert-base-uncased')
    model_bert.to(device)
    
    # Extract class descriptions features once
    print("\nExtracting text features for class descriptions...")
    text_features_list = []
    
    for class_name in CLASS_INDICES:
        description = CLASS_DESCRIPTIONS.get(class_name, f"A {class_name}")
        tokens = tokenizer_bert.tokenize(description)
        tokens = ['[CLS]'] + tokens + ['[SEP]']
        token_ids = tokenizer_bert.convert_tokens_to_ids(tokens)
        
        attention_mask = torch.ones(len(token_ids), dtype=torch.long)
        token_ids = torch.tensor(token_ids).unsqueeze(0).to(device)
        attention_mask = attention_mask.unsqueeze(0).to(device)
        
        with torch.no_grad():
            output_bert = model_bert(token_ids, attention_mask=attention_mask)
            text_feat = output_bert.pooler_output.squeeze(0)
        
        text_features_list.append(text_feat.cpu())
    
    text_features = torch.stack(text_features_list)
    
    # Save text features
    text_features_path = FEATURES_DIR / "Text_features_16.pt"
    torch.save(text_features, text_features_path)
    print(f"✓ Saved text features: {text_features_path} (shape: {text_features.shape})")
    
    # Process each split (Train, Val, Test)
    for split in ['Train', 'Val', 'Test']:
        print(f"\nProcessing {split} split...")
        
        split_path = DATA_ROOT / split
        data_dict = {}
        image_count = 0
        
        for class_name in CLASS_INDICES:
            class_path = split_path / class_name
            
            if not class_path.exists():
                print(f"  Warning: {class_path} does not exist")
                continue
            
            # Find class index
            target_index = CLASS_INDICES.index(class_name)
            
            # Extract features for all images in this class
            for img_name in os.listdir(class_path):
                if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue
                
                img_path = class_path / img_name
                
                try:
                    # Extract image features
                    images = preprocess_clip(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        image_features = model_clip.encode_image(images).squeeze(0)
                        image_features /= image_features.norm()
                    
                    # Store features
                    data_dict[img_name] = {
                        "image_features": image_features.cpu(),
                        "description_embeddings": text_features[target_index],
                        "target_index": target_index
                    }
                    
                    image_count += 1
                except Exception as e:
                    print(f"  Error processing {img_name}: {e}")
            
            print(f"  {class_name}: {image_count} images processed")
        
        # Save split features
        split_output_path = FEATURES_DIR / f"Features_CATALOG_{split.lower()}_16.pt"
        torch.save(data_dict, split_output_path)
        print(f"✓ Saved {split} features: {split_output_path} ({len(data_dict)} images)")

if __name__ == "__main__":
    print("=" * 70)
    print("FEATURE EXTRACTION - New 10-Class Balanced Dataset")
    print("=" * 70)
    
    extract_features()
    
    print("\n" + "=" * 70)
    print("✓ FEATURE EXTRACTION COMPLETE")
    print("=" * 70)
