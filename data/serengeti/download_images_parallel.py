import json
import random
import os
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Load the JSON
with open('SnapshotSerengetiS01.json', 'r') as f:
    data = json.load(f)

# Get categories
categories = {cat['id']: cat['name'] for cat in data['categories']}

# Get images with annotations
image_to_category = {}
for ann in data['annotations']:
    image_id = ann['image_id']
    category_id = ann['category_id']
    if category_id != 0:  # not empty
        image_to_category[image_id] = category_id

# Get image info
images = {img['id']: img for img in data['images']}

# Select 10000 random images with annotations
selected_image_ids = random.sample(list(image_to_category.keys()), 10000)

base_url = 'https://lilawildlife.blob.core.windows.net/lila-wildlife/snapshotserengeti-unzipped/'

# Create directories
os.makedirs('img/Train', exist_ok=True)

for cat_name in categories.values():
    if cat_name != 'empty':
        os.makedirs(f'img/Train/{cat_name}', exist_ok=True)

# Function to download a single image
def download_image(img_id):
    try:
        img_info = images[img_id]
        file_name = img_info['file_name']
        url = base_url + file_name
        cat_id = image_to_category[img_id]
        cat_name = categories[cat_id]
        
        if cat_name == 'empty':
            return None
        
        local_path = f'img/Train/{cat_name}/{os.path.basename(file_name)}'
        
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            with open(local_path, 'wb') as f:
                f.write(response.content)
            return True
        else:
            return False
    except Exception as e:
        return False

# Download with parallel threads (8 concurrent downloads)
print("Starting parallel download with 8 threads...")
downloaded = 0
failed = 0

with ThreadPoolExecutor(max_workers=8) as executor:
    futures = {executor.submit(download_image, img_id): img_id for img_id in selected_image_ids}
    
    with tqdm(total=len(selected_image_ids)) as pbar:
        for future in as_completed(futures):
            result = future.result()
            if result is True:
                downloaded += 1
            elif result is False:
                failed += 1
            pbar.update(1)

print(f'\nDownload complete!')
print(f'Successfully downloaded: {downloaded} images')
print(f'Failed downloads: {failed} images')
print(f'Total: {downloaded + failed} images processed')