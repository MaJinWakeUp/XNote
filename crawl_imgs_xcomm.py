import json
import os
import time
import requests

data_file = "/scratch/jin7/datasets/XCommnunityNote/dataset_4-30.json"
save_dir = "/scratch/jin7/datasets/XCommnunityNote/images"

# read the json file, which contains a list of dictionaries
# For each dict, find the "image_urls" key, which is a list of urls
# For each url, download the image and save it to the save_dir, with the basename of the url

def download_image(url, save_dir):
    # get the basename of the url
    filename = os.path.basename(url)
    # create the save path
    save_path = os.path.join(save_dir, filename)
    if os.path.exists(save_path):
        print(f"{filename} already exists, skipping...")
        return
    # download the image
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(1024):
                f.write(chunk)
        print(f"Downloaded {filename}")
    else:
        print(f"Failed to download {filename}")
    # wait for 1 second
    time.sleep(1)

# read the json file
with open(data_file, 'r') as f:
    data = json.load(f)
# for each dict, find the "image_urls" key, which is a list of urls
for item in data:
    image_urls = item["image_urls"]
    # for each url, download the image and save it to the save_dir, with the basename of the url
    for url in image_urls:
        download_image(url, save_dir)