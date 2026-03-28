import torch
from models.CATALOG_Base import LLaVA_CLIP
import os

def load_features(path):
    return torch.load(path, weights_only=False)

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load test data (same as used for Phase 2)
test_dict = load_features('features/Features_serengeti/standard_features/Features_CATALOG_test_16.pt')
txt_features = load_features('features/Features_serengeti/standard_features/Text_features_16.pt')

test_img = test_dict['image_features'].to(device).float()
test_emb = test_dict['description_embeddings'].to(device).float()
test_labels = test_dict['target_index'].to(device)
txt_global = txt_features.to(device).float()

print(f"Test data (same as Phase 2): {test_img.shape}, {test_emb.shape}")
print(f"Test labels distribution: {[int((test_labels==i).sum()) for i in range(10)]}")

# Find latest Phase 1 model
phase1_base = 'Best/exp_Base_Out_domain'
if os.path.exists(phase1_base):
    dirs = sorted([d for d in os.listdir(phase1_base) if d.startswith('training_')])
    if dirs:
        latest_dir = os.path.join(phase1_base, dirs[-1])
        model_file = [f for f in os.listdir(latest_dir) if f.endswith('.pth')]
        if model_file:
            model_path = os.path.join(latest_dir, model_file[0])
            print(f"\nPhase 1 model: {model_path}")
            
            # Test Phase 1
            model = LLaVA_CLIP(hidden_dim=1045, num_layers=1, dropout=0.381, device=device, num_classes=10)
            model.to(device)
            model.load_state_dict(torch.load(model_path, weights_only=False))
            model.eval()
            
            correct = 0
            with torch.no_grad():
                for i in range(0, len(test_img), 26):
                    batch_img = test_img[i:i+26]
                    batch_emb = test_emb[i:i+26]
                    batch_labels = test_labels[i:i+26]
                    _, acc, _ = model.predict(batch_emb, batch_img, txt_global, 0.60855, batch_labels, 0.1)
                    correct += acc.item()
            
            phase1_acc = correct / len(test_img) * 100
            print(f"Phase 1 Test Accuracy: {phase1_acc:.2f}%")
            print(f"(This should be ~97.13% if working correctly)")
