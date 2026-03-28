import os
import random
import shutil
from tqdm import tqdm

# Paths
source_dir = 'img/Train'
val_dest = 'img/Val'
test_dest = 'img/Test'

# Create Val and Test directories
os.makedirs(val_dest, exist_ok=True)
os.makedirs(test_dest, exist_ok=True)

# Get all categories
all_categories = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]

print(f"Found {len(all_categories)} categories")
print("Splitting 10,000 images into: 6,000 Train, 2,000 Val, 2,000 Test (60/20/20)")

# For each category, split 60/20/20 (train/val/test)
for category in tqdm(all_categories):
    category_path = os.path.join(source_dir, category)
    images = os.listdir(category_path)
    
    total_images = len(images)
    val_count = max(1, int(total_images * 0.2))    # 20% for val
    test_count = max(1, int(total_images * 0.2))   # 20% for test
    
    # Randomly select images
    val_images = random.sample(images, min(val_count, total_images))
    remaining = [img for img in images if img not in val_images]
    test_images = random.sample(remaining, min(test_count, len(remaining)))
    
    # Create category directories in val and test folders
    val_category_path = os.path.join(val_dest, category)
    test_category_path = os.path.join(test_dest, category)
    os.makedirs(val_category_path, exist_ok=True)
    os.makedirs(test_category_path, exist_ok=True)
    
    # Move val images
    for img in val_images:
        src = os.path.join(category_path, img)
        dst = os.path.join(val_category_path, img)
        shutil.move(src, dst)
    
    # Move test images
    for img in test_images:
        src = os.path.join(category_path, img)
        dst = os.path.join(test_category_path, img)
        shutil.move(src, dst)

print("\nSplit complete!")

# Count files
train_count = sum(len(os.listdir(os.path.join(source_dir, cat))) for cat in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, cat)))
val_count = sum(len(os.listdir(os.path.join(val_dest, cat))) for cat in os.listdir(val_dest) if os.path.isdir(os.path.join(val_dest, cat)))
test_count = sum(len(os.listdir(os.path.join(test_dest, cat))) for cat in os.listdir(test_dest) if os.path.isdir(os.path.join(test_dest, cat)))

print(f"Train images: {train_count}")
print(f"Val images: {val_count}")
print(f"Test images: {test_count}")
print(f"Total: {train_count + val_count + test_count}")