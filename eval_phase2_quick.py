#!/usr/bin/env python
"""Quick Phase 2 evaluation - summary only"""
import torch
import torch.nn as nn
from models.CATALOG_Base_Phase2 import LLaVA_CLIP_Phase2

def load_features(feature_path):
    return torch.load(feature_path, weights_only=False)

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load test data
test_dict = load_features('features/Features_serengeti/standard_features/Features_CATALOG_test_16.pt')
text_features = load_features('features/Features_serengeti/standard_features/Text_features_16.pt')

test_img = test_dict['image_features'].to(device).float()
test_txt_emb = test_dict['description_embeddings'].to(device).float()
test_labels = test_dict['target_index'].to(device)
txt_global = text_features.to(device).float()

# Load model
model = LLaVA_CLIP_Phase2(
    hidden_dim=1743, num_layers=4, dropout=0.381, device=device, 
    num_classes=10, enable_classifier_fusion=True
)
model.to(device)
model_path = 'Best/exp_Base_In_domain_Serengeti_Phase2/training_2026-03-28_17-33-05/best_model_params_4_1743.pth'
model.load_state_dict(torch.load(model_path, weights_only=False))
model.eval()

# Evaluate
test_correct = 0
batch_size = 26
with torch.no_grad():
    for i in range(0, len(test_img), batch_size):
        batch_img = test_img[i:i+batch_size]
        batch_emb = test_txt_emb[i:i+batch_size]
        batch_labels = test_labels[i:i+batch_size]
        _, acc, _ = model.predict(batch_emb, batch_img, txt_global, 0.60855, batch_labels, 0.1)
        test_correct += acc.item()
        if (i // batch_size) % 20 == 0:
            print(f"Batch {i//batch_size}/{len(test_img)//batch_size}...", flush=True)

test_acc = test_correct / len(test_img) * 100
print(f"\n{'='*50}")
print(f"PHASE 2 TEST ACCURACY: {test_acc:.2f}%")
print(f"{'='*50}")
