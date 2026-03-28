import torch
from models.CATALOG_Base_Phase2 import LLaVA_CLIP_Phase2

def load_features(path):
    return torch.load(path, weights_only=False)

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load data
test_dict = load_features('features/Features_serengeti/standard_features/Features_CATALOG_test_16.pt')
txt_features = load_features('features/Features_serengeti/standard_features/Text_features_16.pt')

test_img = test_dict['image_features'].to(device).float()
test_emb = test_dict['description_embeddings'].to(device).float()
test_labels = test_dict['target_index'].to(device)
txt_global = txt_features.to(device).float()

# Load model
model = LLaVA_CLIP_Phase2(hidden_dim=1743, num_layers=4, dropout=0.381, device=device, num_classes=10)
model.to(device)
state = torch.load('Best/exp_Base_In_domain_Serengeti_Phase2/training_2026-03-28_17-33-05/best_model_params_4_1743.pth', weights_only=False)
model.load_state_dict(state)
model.eval()

# Get all predictions
all_preds = []
all_labels = []
batch_size = 26

with torch.no_grad():
    for i in range(0, len(test_img), batch_size):
        batch_img = test_img[i:i+batch_size]
        batch_emb = test_emb[i:i+batch_size]
        batch_labels = test_labels[i:i+batch_size]
        
        _, _, preds = model.predict(batch_emb, batch_img, txt_global, 0.60855, batch_labels, 0.1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch_labels.cpu().numpy())

# Analyze predictions
import numpy as np
all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

print(f"Prediction distribution:")
for c in range(10):
    pred_count = (all_preds == c).sum()
    true_count = (all_labels == c).sum()
    acc = ((all_preds == c) & (all_preds == all_labels)).sum() / max(true_count, 1) * 100
    print(f"  Class {c}: Predicted={pred_count}, Actual={true_count}, Acc={acc:.1f}%")

# Check if model is just predicting same class
unique_preds = np.unique(all_preds)
print(f"\nUnique predictions made: {unique_preds}")

# Calculate actual accuracy
correct = (all_preds == all_labels).sum()
total = len(all_labels)
print(f"\nActual test accuracy: {correct}/{total} = {correct/total*100:.2f}%")
