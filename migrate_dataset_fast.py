"""
Optimized dataset migration script using xcopy for faster transfers
"""

import os
import subprocess
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
    
    total_classes = len(NEW_CLASSES)
    for idx, class_name in enumerate(NEW_CLASSES, 1):
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
        
        # Create temporary subdirectory for this class in source
        splits_temp = {
            'Train': train_images,
            'Val': val_images,
            'Test': test_images
        }
        
        # Copy each split
        for split_name, image_list in splits_temp.items():
            dst_dir = os.path.join(TARGET_PATH, split_name, class_name)
            
            for img in image_list:
                src = os.path.join(source_dir, img)
                dst = os.path.join(dst_dir, img)
                
                try:
                    with open(src, 'rb') as src_file:
                        with open(dst, 'wb') as dst_file:
                            dst_file.write(src_file.read())
                except Exception as e:
                    print(f"  Error copying {img}: {e}")
        
        print(f"  [{idx}/{total_classes}] {class_name}: Train={len(train_images)}, Val={len(val_images)}, Test={len(test_images)}")
    
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
            if os.path.exists(class_path):
                count = len([f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))])
                total_images += count
        print(f"  {split}: {total_images} images across {len(NEW_CLASSES)} classes")
    
    print("✓ Migration complete!")

if __name__ == "__main__":
    print("=" * 60)
    print("DATASET MIGRATION SCRIPT (Optimized)")
    print("=" * 60)
    
    create_split_directories()
    copy_images_with_split()
    update_feature_extraction_script()
    verify_results()
    
    print("\n" + "=" * 60)
    print("Next steps:")
    print("1. Run feature extraction: python feature_extraction/Base/CATALOG_extraction_features_serengeti.py")
    print("2. Retrain the model with improved config")
    print("=" * 60)
