"""
Dataset migration script:
1. Delete old Serengeti images from data/serengeti/img/Train, Test, Val
2. Copy new wilddata images with train/test/val splits (70/15/15)
3. Update feature extraction script for new 10 classes
"""

import os
import shutil
import random
from pathlib import Path

# Configuration
WILDDATA_PATH = r"C:\Users\rishi\OneDrive\Documents\WildDataset\serengeti_reduced"
TARGET_PATH = r"C:\Users\rishi\CATALOG\data\serengeti\img"
FEATURE_EXTRACTION_SCRIPT = r"C:\Users\rishi\CATALOG\feature_extraction\Base\CATALOG_extraction_features_serengeti.py"

# New 10 classes
NEW_CLASSES = [
    'elephant', 'gazellegrants', 'gazellethomsons', 'giraffe', 'guineafowl',
    'hartebeest', 'hyenaspotted', 'lionfemale', 'warthog', 'zebra'
]

# Train/Val/Test split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

def delete_old_images():
    """Delete old Serengeti images"""
    print("Deleting old Serengeti images...")
    for split in ['Train', 'Test', 'Val']:
        split_path = os.path.join(TARGET_PATH, split)
        if os.path.exists(split_path):
            # Remove only class subdirectories, keep the folder
            for class_folder in os.listdir(split_path):
                class_path = os.path.join(split_path, class_folder)
                if os.path.isdir(class_path):
                    print(f"  Removing {class_path}")
                    shutil.rmtree(class_path)
    print("✓ Old images deleted")

def create_split_directories():
    """Create train/test/val directories for new classes"""
    print("\nCreating directory structure...")
    for split in ['Train', 'Test', 'Val']:
        split_path = os.path.join(TARGET_PATH, split)
        os.makedirs(split_path, exist_ok=True)
        
        for class_name in NEW_CLASSES:
            class_dir = os.path.join(split_path, class_name)
            os.makedirs(class_dir, exist_ok=True)
    print("✓ Directory structure created")

def copy_images_with_split():
    """Copy images from wilddata with train/test/val split"""
    print("\nCopying images with train/test/val split...")
    
    for class_name in NEW_CLASSES:
        source_dir = os.path.join(WILDDATA_PATH, class_name)
        
        # Get all image files
        images = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]
        images.sort()
        random.shuffle(images)
        
        # Calculate split indices
        total = len(images)
        train_count = int(total * TRAIN_RATIO)
        val_count = int(total * VAL_RATIO)
        
        train_images = images[:train_count]
        val_images = images[train_count:train_count + val_count]
        test_images = images[train_count + val_count:]
        
        # Copy to Train
        for img in train_images:
            src = os.path.join(source_dir, img)
            dst = os.path.join(TARGET_PATH, 'Train', class_name, img)
            shutil.copy2(src, dst)
        
        # Copy to Val
        for img in val_images:
            src = os.path.join(source_dir, img)
            dst = os.path.join(TARGET_PATH, 'Val', class_name, img)
            shutil.copy2(src, dst)
        
        # Copy to Test
        for img in test_images:
            src = os.path.join(source_dir, img)
            dst = os.path.join(TARGET_PATH, 'Test', class_name, img)
            shutil.copy2(src, dst)
        
        print(f"  {class_name}: Train={len(train_images)}, Val={len(val_images)}, Test={len(test_images)}")
    
    print("✓ Images copied with splits")

def update_feature_extraction_script():
    """Update the feature extraction script with new 10 classes"""
    print("\nUpdating feature extraction script...")
    
    with open(FEATURE_EXTRACTION_SCRIPT, 'r') as f:
        content = f.read()
    
    # Find the class_indices section and replace it
    old_class_list = """class_indices=['aardvark', 'aardwolf', 'baboon', 'batEaredFox', 'buffalo', 'bushbuck', 'caracal', 'cheetah', 'civet',
             'dikDik',
             'eland', 'elephant', 'gazelleGrants', 'gazelleThomsons', 'genet', 'giraffe', 'guineaFowl', 'hare','hartebeest', 'hippopotamus','honeyBadger', 'hyenaSpotted', 'hyenaStriped', 'impala', 'jackal', 'koriBustard', 'leopard', 'lionFemale',
             'lionMale', 'mongoose','ostrich', 'porcupine', 'reedbuck', 'reptiles', 'rhinoceros', 'rodents', 'secretaryBird', 'serval', 'topi', 'vervetMonkey','warthog', 'waterbuck', 'wildcat', 'wildebeest', 'zebra', 'zorilla']"""
    
    new_class_list = """class_indices=['elephant', 'gazellegrants', 'gazellethomsons', 'giraffe', 'guineafowl',
             'hartebeest', 'hyenaspotted', 'lionfemale', 'warthog', 'zebra']"""
    
    if old_class_list in content:
        content = content.replace(old_class_list, new_class_list)
        with open(FEATURE_EXTRACTION_SCRIPT, 'w') as f:
            f.write(content)
        print(f"✓ Feature extraction script updated with 10 new classes")
    else:
        print("⚠ Warning: Could not find exact class_indices match in feature extraction script")
        print("  Please manually verify the class_indices in the script")

def verify_results():
    """Verify the migration"""
    print("\nVerifying migration...")
    
    for split in ['Train', 'Test', 'Val']:
        split_path = os.path.join(TARGET_PATH, split)
        total_images = 0
        for class_name in NEW_CLASSES:
            class_path = os.path.join(split_path, class_name)
            count = len([f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))])
            total_images += count
        print(f"  {split}: {total_images} images across {len(NEW_CLASSES)} classes")
    
    print("✓ Migration complete!")

if __name__ == "__main__":
    print("=" * 60)
    print("DATASET MIGRATION SCRIPT")
    print("=" * 60)
    
    # Confirm before deletion
    response = input("\n⚠ This will DELETE all old images in data/serengeti/img/Train, Test, Val\nContinue? (yes/no): ")
    if response.lower() not in ['yes', 'y']:
        print("Abort.")
        exit(1)
    
    delete_old_images()
    create_split_directories()
    copy_images_with_split()
    update_feature_extraction_script()
    verify_results()
    
    print("\n" + "=" * 60)
    print("Next steps:")
    print("1. Run feature extraction: python feature_extraction/Base/CATALOG_extraction_features_serengeti.py")
    print("2. Retrain the model with improved config")
    print("=" * 60)
