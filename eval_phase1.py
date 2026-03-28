#!/usr/bin/env python
"""Test Phase 1 baseline model for comparison"""
import torch
import torch.nn as nn
from models.CATALOG_Base import LLaVA_CLIP
import os

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

print(f"Test data: {test_img.shape}, {test_txt_emb.shape}, Labels: {test_labels.unique()}")

# Find Phase 1 model
phase1_dirs = [d for d in os.listdir('Best/exp_Base_Out_domain') if d.startswith('training_')]
if phase1_dirs:
    phase1_dir = os.path.join('Best/exp_Base_Out_domain', sorted(phase1_dirs)[-1])
    model_files = [f for f in os.listdir(phase1_dir) if f.endswith('.pth')]
    if model_files:
        model_path = os.path.join(phase1_dir, model_files[0])
        print(f"Found Phase 1 model: {model_path}")
        
        # Load model
        model = LLaVA_CLIP(hidden_dim=1045, num_layers=1, dropout=0.381, device=device, num_classes=10)
        model.to(device)
        state = torch.load(model_path, weights_only=False)
        model.load_state_dict(state)
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
        
        test_acc = test_correct / len(test_img) * 100
        print(f"\nPhase 1 Test Accuracy: {test_acc:.2f}%")
    else:
        print("No model files found in Phase 1 directory")
else:
    print("No Phase 1 training directories found")
