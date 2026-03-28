import torch

# Load all datasets
train = torch.load('features/Features_serengeti/standard_features/Features_CATALOG_train_16.pt', weights_only=False)
val = torch.load('features/Features_serengeti/standard_features/Features_CATALOG_val_16.pt', weights_only=False)
test = torch.load('features/Features_serengeti/standard_features/Features_CATALOG_test_16.pt', weights_only=False)

print(f'Train size: {len(train["target_index"])}')
print(f'Val size: {len(val["target_index"])}')
print(f'Test size: {len(test["target_index"])}')
print(f'Total: {len(train["target_index"]) + len(val["target_index"]) + len(test["target_index"])}')

# Check label distributions
print(f'\nTrain labels: {sorted(train["target_index"].unique().tolist())}')
print(f'Test labels: {sorted(test["target_index"].unique().tolist())}')

# Check class balance
print('\nLabel distribution:')
for c in range(10):
    tr = (train['target_index'] == c).sum().item()
    te = (test['target_index'] == c).sum().item()
    v = (val['target_index'] == c).sum().item()
    print(f'  Class {c}: Train={tr}, Val={v}, Test={te}')

# Check feature distribution
print(f'\nFeature ranges:')
print(f'  Train: min={train["image_features"].min():.4f}, max={train["image_features"].max():.4f}')
print(f'  Test: min={test["image_features"].min():.4f}, max={test["image_features"].max():.4f}')
