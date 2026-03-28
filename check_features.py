import os

feat_dir = r'C:\Users\rishi\CATALOG\features\Features_serengeti\standard_features'
pt_files = sorted([f for f in os.listdir(feat_dir) if f.endswith('.pt')])

print(f"\n✓ Found {len(pt_files)} .pt files:\n")
for f in pt_files:
    size_mb = os.path.getsize(os.path.join(feat_dir, f)) / (1024*1024)
    print(f"  {f}: {size_mb:.2f} MB")
