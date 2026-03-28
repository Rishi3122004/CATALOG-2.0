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

print(f"Data shapes: Img={test_img.shape}, Emb={test_emb.shape}, Txt={txt_global.shape}")

# Load model
model = LLaVA_CLIP_Phase2(hidden_dim=1743, num_layers=4, dropout=0.381, device=device, num_classes=10)
model.to(device)
state = torch.load('Best/exp_Base_In_domain_Serengeti_Phase2/training_2026-03-28_17-33-05/best_model_params_4_1743.pth', weights_only=False)
model.load_state_dict(state)

# Test BOTH training and eval mode
for mode_name, eval_flag in [("TRAINING MODE", False), ("EVAL MODE", True)]:
    if eval_flag:
        model.eval()
    else:
        model.train()
    
    print(f"\n{'='*50}")
    print(f"Testing in {mode_name}")
    print(f"{'='*50}")
    
    correct = 0
    batch_size = 26
    
    with torch.no_grad():
        # Test first batch separately to debug
        batch_img = test_img[:batch_size]
        batch_emb = test_emb[:batch_size]
        batch_labels = test_labels[:batch_size]
        
        print(f"First batch shapes: img={batch_img.shape}, emb={batch_emb.shape}")
        
        # Get predictions
        loss, acc_count, preds = model.predict(batch_emb, batch_img, txt_global, 0.60855, batch_labels, 0.1)
        
        print(f"Batch 0: Loss={loss.item():.4f}, Correct={acc_count.item()}/{len(batch_labels)}")
        print(f"Predictions: {preds[:5]}, Labels: {batch_labels[:5]}")
        print(f"Accuracy this batch: {acc_count.item()/len(batch_labels)*100:.2f}%")
        
        # Full evaluation
        for i in range(0, len(test_img), batch_size):
            batch_img = test_img[i:i+batch_size]
            batch_emb = test_emb[i:i+batch_size]
            batch_labels = test_labels[i:i+batch_size]
            
            _, acc, _ = model.predict(batch_emb, batch_img, txt_global, 0.60855, batch_labels, 0.1)
            correct += acc.item()
        
        test_acc = correct / len(test_img) * 100
        print(f"Overall accuracy: {test_acc:.2f}%")
