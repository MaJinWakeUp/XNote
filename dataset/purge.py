import os
import json

# Paths
dataset_path = 'final_dataset.json'
evidence_dir ='./evidence/'
gcloud_dir = './gcloud_search/'
reverse_dir = './reverse_search/'
summarized_evidence_dir = './summarized_evidence/'
images_dir = './images/'

# Load dataset
with open(dataset_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Collect all valid ids and images
valid_ids = set()
valid_images = set()
for item in data:
    if 'id' in item:
        valid_ids.add(str(item['id']))
    if 'images' in item:
        valid_images.update(item['images'])

# Helper to remove files not in valid set
def purge_dir(directory, valid_set):
    if not os.path.isdir(directory):
        return
    for fname in os.listdir(directory):
        fpath = os.path.join(directory, fname)
        # Remove extension for id comparison
        file_id = os.path.splitext(fname)[0]
        if file_id not in valid_set:
            os.remove(fpath)
            print(f"Removed: {fpath}")

# Purge evidence, gcloud_search, reverse_search by id
for d in [evidence_dir, gcloud_dir, reverse_dir, summarized_evidence_dir]:
    purge_dir(d, valid_ids)

# Purge images by image filename
if os.path.isdir(images_dir):
    for fname in os.listdir(images_dir):
        fpath = os.path.join(images_dir, fname)
        if fname not in valid_images:
            os.remove(fpath)
            print(f"Removed: {fpath}")