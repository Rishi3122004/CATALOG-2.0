#!/usr/bin/env python
"""Quick check: Are description embeddings really all zeros?"""

import torch
import numpy as np
from pathlib import Path

# Load train data
train_data = torch.load('features/Features_serengeti/standard_features/Features_CATALOG_train_16.pt', weights_only=False)

desc_emb = train_data['description_embeddings']
img_feat = train_data['image_features']
text_feat = torch.load('features/Features_serengeti/standard_features/Text_features_16.pt', weights_only=False)

print("Description Embeddings Stats:")
print(f"  Unique values: {len(torch.unique(desc_emb))}")
print(f"  All zeros? {torch.all(desc_emb == 0).item()}")
print(f"  Sample values:\n{desc_emb[0, :10]}")

print("\nImage Features Stats:")
print(f"  Unique values: {len(torch.unique(img_feat))}")
print(f"  Sample values:\n{img_feat[0, :10]}")

print("\nText Features Stats:")
print(f"  Shape: {text_feat.shape}")
print(f"  Unique values: {len(torch.unique(text_feat))}")
print(f"  Sample values:\n{text_feat[:, 0]}")

# Check if mixing them
print("\nFusioning test (as in model):")
sim_clip = img_feat[:10] @ text_feat.t()
sim_bert = desc_emb[:10] @ text_feat.t()

print(f"  CLIP similarity: {sim_clip[0, :3]}")
print(f"  BERT similarity (all zeros?): {sim_bert[0, :3]}")

weight_clip = 0.60855
fused = sim_clip[:10] * weight_clip + sim_bert[:10] * (1 - weight_clip)
print(f"  Fused (should be mostly CLIP): {fused[0, :3]}")
