import torch
from models.CATALOG_Base_Phase2 import LLaVA_CLIP_Phase2

def load_features(path):
    return torch.load(path, weights_only=False)

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load all datasets
train_dict = load_features('features/Features_serengeti/standard_features/Features_CATALOG_train_16.pt')
val_dict = load_features('features/Features_serengeti/standard_features/Features_CATALOG_val_16.pt')
test_dict = load_features('features/Features_serengeti/standard_features/Features_CATALOG_test_16.pt')
txt_features = load_features('features/Features_serengeti/standard_features/Text_features_16.pt')

txt_global = txt_features.to(device).float()

datasets = {
    'TRAIN': (train_dict['image_features'], train_dict['description_embeddings'], train_dict['target_index']),
    'VAL': (val_dict['image_features'], val_dict['description_embeddings'], val_dict['target_index']),
    'TEST': (test_dict['image_features'], test_dict['description_embeddings'], test_dict['target_index']),
}

# Load trained model
model = LLaVA_CLIP_Phase2(hidden_dim=1743, num_layers=4, dropout=0.381, device=device, num_classes=10)
model.to(device)
state = torch.load('Best/exp_Base_In_domain_Serengeti_Phase2/training_2026-03-28_17-33-05/best_model_params_4_1743.pth', weights_only=False)
model.load_state_dict(state)
model.eval()

print(f"Testing Phase 2 model on all datasets:")
print(f"{'Dataset':<10} {'Accuracy':<12} {'Samples':<10}")
print(f"{'-'*40}")

for dataset_name, (img, emb, labels) in datasets.items():
    img = img.to(device).float()
    emb = emb.to(device).float()
    labels = labels.to(device)
    
    correct = 0
    with torch.no_grad():
        for i in range(0, len(img), 26):
            batch_img = img[i:i+26]
            batch_emb = emb[i:i+26]
            batch_labels = labels[i:i+26]
            _, acc, _ = model.predict(batch_emb, batch_img, txt_global, 0.60855, batch_labels, 0.1)
            correct += acc.item()
    
    acc_pct = correct / len(img) * 100
    print(f"{dataset_name:<10} {acc_pct:>10.2f}%  {len(img):>8} samples")

print(f"{'-'*40}")
print(f"\nAnalysis:")
print(f"  If TRAIN > TEST: Model memorized training data")
print(f"  If TRAIN ≈ VAL ≈ TEST: Model has consistent generalization")
print(f"  If TEST >> VAL: Suggests data leakage or distribution mismatch")
