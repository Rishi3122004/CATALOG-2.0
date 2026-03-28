import torch
import torch.nn as nn
from models.CATALOG_Base_Phase2 import LLaVA_CLIP_Phase2
import os

def load_features(feature_path):
    if os.path.exists(feature_path):
        return torch.load(feature_path, weights_only=False)
    else:
        print(f"File not found: {feature_path}")
        return None

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load features
print("[LOADING] Test features...")
test_dict = load_features('features/Features_serengeti/standard_features/Features_CATALOG_test_16.pt')
text_features_global = load_features('features/Features_serengeti/standard_features/Text_features_16.pt')

test_img = test_dict['image_features'].to(device).float()
test_txt_emb = test_dict['description_embeddings'].to(device).float()
test_labels = test_dict['target_index'].to(device)
txt_global = text_features_global.to(device).float()  # Direct tensor, not dictionary

print(f"Test image features: {test_img.shape}")
print(f"Test text embeddings: {test_txt_emb.shape}")
print(f"Global text features: {txt_global.shape}")

# Load best model
model_path = 'Best/exp_Base_In_domain_Serengeti_Phase2/training_2026-03-28_17-33-05/best_model_params_4_1743.pth'
config = {
    'hidden_dim': 1743,
    'num_layers': 4,
    'dropout': 0.381,
    'num_classes': 10,
    'batch_size': 26,
    'weight_Clip': 0.60855,
    't': 0.1,
}

print(f"\n[MODEL] Loading Phase 2 model from {model_path}...")
model = LLaVA_CLIP_Phase2(
    hidden_dim=config['hidden_dim'],
    num_layers=config['num_layers'],
    dropout=config['dropout'],
    device=device,
    num_classes=config['num_classes'],
    enable_classifier_fusion=True
)
model.to(device)
model.load_state_dict(torch.load(model_path, weights_only=False))
model.eval()

# Test
print("\n[TEST] Evaluating on test set...")
test_loss = 0.0
test_correct = 0
batch_count = 0
all_preds = []
all_targets = []

with torch.no_grad():
    for i in range(0, len(test_img), config['batch_size']):
        batch_img = test_img[i:i+config['batch_size']]
        batch_emb = test_txt_emb[i:i+config['batch_size']]
        batch_labels = test_labels[i:i+config['batch_size']]
        
        loss, acc, preds = model.predict(batch_emb, batch_img, txt_global, 
                                        config['weight_Clip'], batch_labels, config['t'])
        
        test_loss += loss.item()
        test_correct += acc.item()
        batch_count += 1
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(batch_labels.cpu().numpy())

test_loss_avg = test_loss / batch_count
test_acc = test_correct / len(test_img) * 100

print(f"\nFinal Test Loss: {test_loss_avg:.4f}")
print(f"Final Test Accuracy: {test_acc:.4f}%")

# Per-class accuracy
print("\nPer-class accuracy:")
for cls in range(10):
    cls_mask = [t == cls for t in all_targets]
    if sum(cls_mask) > 0:
        cls_acc = sum([p == t for p, t, m in zip(all_preds, all_targets, cls_mask) if m]) / sum(cls_mask) * 100
        print(f"  Class {cls}: {cls_acc:.2f}%")
