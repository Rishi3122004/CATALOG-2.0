import torch
from models.CATALOG_Base_Phase2 import LLaVA_CLIP_Phase2

def load_features(path):
    return torch.load(path, weights_only=False)

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load test data
test_dict = load_features('features/Features_serengeti/standard_features/Features_CATALOG_test_16.pt')
txt_features = load_features('features/Features_serengeti/standard_features/Text_features_16.pt')

test_img = test_dict['image_features'].to(device).float()
test_emb = test_dict['description_embeddings'].to(device).float()
test_labels = test_dict['target_index'].to(device)
txt_global = txt_features.to(device).float()

print("Testing with RANDOM (untrained) Phase 2 model:")
print("="*60)

# Create RANDOM model (no loading)
model_random = LLaVA_CLIP_Phase2(hidden_dim=1743, num_layers=4, dropout=0.381, device=device, num_classes=10)
model_random.to(device)
model_random.eval()

correct_random = 0
with torch.no_grad():
    for i in range(0, len(test_img), 26):
        batch_img = test_img[i:i+26]
        batch_emb = test_emb[i:i+26]
        batch_labels = test_labels[i:i+26]
        _, acc, _ = model_random.predict(batch_emb, batch_img, txt_global, 0.60855, batch_labels, 0.1)
        correct_random += acc.item()

random_acc = correct_random / len(test_img) * 100
print(f"Random model accuracy: {random_acc:.2f}%")

print("\nTesting with TRAINED Phase 2 model:")
print("="*60)

# Create model and load weights
model_trained = LLaVA_CLIP_Phase2(hidden_dim=1743, num_layers=4, dropout=0.381, device=device, num_classes=10)
model_trained.to(device)
state = torch.load('Best/exp_Base_In_domain_Serengeti_Phase2/training_2026-03-28_17-33-05/best_model_params_4_1743.pth', weights_only=False)
model_trained.load_state_dict(state)
model_trained.eval()

correct_trained = 0
with torch.no_grad():
    for i in range(0, len(test_img), 26):
        batch_img = test_img[i:i+26]
        batch_emb = test_emb[i:i+26]
        batch_labels = test_labels[i:i+26]
        _, acc, _ = model_trained.predict(batch_emb, batch_img, txt_global, 0.60855, batch_labels, 0.1)
        correct_trained += acc.item()

trained_acc = correct_trained / len(test_img) * 100
print(f"Trained model accuracy: {trained_acc:.2f}%")

print(f"\n{'='*60}")
print(f"Difference: {trained_acc - random_acc:.2f}%")
if trained_acc - random_acc < 5:
    print("⚠️  WARNING: Very small difference suggests model weights not loaded correctly!")
