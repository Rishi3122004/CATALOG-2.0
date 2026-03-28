import json
import random
import os
import requests
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

# Download images
for img_id in tqdm(selected_image_ids):
    img_info = images[img_id]
    file_name = img_info['file_name']
    url = base_url + file_name
    cat_id = image_to_category[img_id]
    cat_name = categories[cat_id]
    if cat_name == 'empty':
        continue
    local_path = f'img/Train/{cat_name}/{os.path.basename(file_name)}'
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            with open(local_path, 'wb') as f:
                f.write(response.content)
        else:
            print(f'Failed to download {url}')
    except Exception as e:
        print(f'Error downloading {url}: {e}')

print('Downloaded 10000 images.')