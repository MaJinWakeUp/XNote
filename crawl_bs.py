import os
import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import json
import time

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:93.0) Gecko/20100101 Firefox/93.0",
}

def is_accessible(url):
    try:
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            return False
        else:
            return response
    except requests.RequestException:
        return False

def parse_content(content, url):
    print(content)
    if "perma.cc" in url:
        meta_description = content.find('div', attrs={"data-testid": "post_message"})
        if meta_description:
            caption = meta_description.get('content')
        else:
            caption_tag = content.find('span', class_='d2edcug0 hpfvmrgz qv66sw1b c1et5uql gk29lw5a a8c37x1j keod5gw0 nxhoafnm aigsh9s9 d9wwppkn fe6kdd0r mau55g9w c8b282yb hrzyx87i jq4qci2q a3bd9o3v knj5qynh oo9gr5id')
            if caption_tag:
                caption = caption_tag.text
            else:
                caption = ""
        time_tag = content.find('p', class_='creation')
        if time_tag:
            time = time_tag.text
        else:
            time = ""
    elif "archive.ph" in url or "archive.is" in url or "archive.vn" in url:
        meta_description = content.find('meta', property='og:title')
        if meta_description:
            caption = meta_description.get('content')
        else:
            caption = ""
        time_tag = content.find('time', itemprop='pubdate')
        if time_tag:
            time = time_tag.text
        else:
            time = ""
    elif "facebook.com" in url:
        meta_description = content.find('meta', property='description')
        if meta_description:
            caption = meta_description.get('content')
        else:
            caption = ""
        time_tag = content.find('a', tabindex='0')
        if time_tag:
            time = time_tag.text
        else:
            time = ""
    elif "twitter.com" in url:
        meta_description = content.find('meta', property='og:title')
        if meta_description:
            caption = meta_description.get('content')
        else:
            caption = ""
        time_tag = content.find('time')
        if time_tag:
            time = time_tag.text
        else:
            time = ""
    elif "instagram.com" in url:
        meta_description = content.find('meta', property='og:title')
        if meta_description:
            caption = meta_description.get('content')
        else:
            caption = ""
        time_tag = content.find('time')
        if time_tag:
            time = time_tag.text
        else:
            time = ""
    elif "imgur.com" in url:
        meta_description = content.find('h1')
        if meta_description:
            caption = meta_description.text
        else:
            caption = ""
        # get the span with title containing "Time" or "time"
        time_tag = content.find('span', title=lambda x: x and 'time' in x.lower())
        if time_tag:
            time = time_tag.text
        else:
            time = ""
    elif "reddit.com" in url:
        meta_description = content.find('h1')
        if meta_description:
            caption = meta_description.text
        else:
            caption = ""
        time_tag = content.find_all('time')
        if time_tag:
            time = time_tag[0].get('title')
        else:
            time = ""
    elif "evernote.com" in url:
        meta_description = content.find('div', attrs={"data-testid": "view-only-title"})
        if meta_description:
            caption = meta_description.text
        else:
            caption = ""
        time = ""
    elif "web.archive" in url:
        meta_description = content.find('meta', property='og:description')
        if meta_description:
            caption = meta_description.get('content')
        else:
            caption = ""
        time_tag = content.find('time')
        if time_tag:
            time = time_tag.text
        else:
            time = ""
    else:
        caption = ""
        time = ""
    return caption, time

# test_url = "https://perma.cc/75LH-32FH?view-mode=server-side"
# # test_url = "https://perma.cc/8YLW-72CF?view-mode=client-side"
# response = is_accessible(test_url)
# soup = BeautifulSoup(response.content, 'html.parser')
# caption, time = parse_content(soup, test_url)
# print(caption, time)
# exit()

def find_image_url(image_meta, image_id):
    image_url = image_meta[image_meta['image_id'] == image_id]['url'].values[0]
    return image_url


image_meta_file = '/scratch/jin7/datasets/AMMeBa/image_metadata.csv'
stage2_file = '/scratch/jin7/datasets/AMMeBa/stage_2.csv'
image_save_dir = '/scratch/jin7/datasets/AMMeBa/images_part1'
if not os.path.exists(image_save_dir):
    os.makedirs(image_save_dir)

image_meta_df = pd.read_csv(image_meta_file)
stage2_df = pd.read_csv(stage2_file)

ooc_data = []
all_sources = []
all_imids = []
for index, row in tqdm(stage2_df.head(10000).iterrows(), total=min(10000, stage2_df.shape[0])):
    if str(row['disqualified']).lower() == 'true':
        continue
    else:
        bi = str(row['context_manipulation']).lower()
        image_id = row['image_id']
        image_url = find_image_url(image_meta_df, image_id)
        if bi == 'true':
            response_source = is_accessible(row['misinfo_source'])
            response_image = is_accessible(image_url)
            if response_source and response_image:
                if 'Content-Type' in response_image.headers and response_image.headers['Content-Type'].startswith('image/'):
                    if image_id in all_imids and row['misinfo_source'] in all_sources:
                        continue
                    # soup_source = BeautifulSoup(response_source.content, 'html.parser')
                    # caption, time = parse_content(soup_source, row['misinfo_source'])
                    # if caption == "" and time == "":
                        # continue
                    # add self_miscontextualizing,circumstance_manipulation,identity_manipulation,location_manipulation,datetime_manipulation,atypical_manipulation
                    cur_data = {
                        'misinfo_source': row['misinfo_source'],
                        'image_id': image_id,
                        'caption': "",
                        'platform': "",
                        'time': row['submission_time'],
                        'self_miscontextualizing': row['self_miscontextualizing'],
                        'circumstance_manipulation': row['circumstance_manipulation'],
                        'identity_manipulation': row['identity_manipulation'],
                        'location_manipulation': row['location_manipulation'],
                        'datetime_manipulation': row['datetime_manipulation'],
                        'atypical_manipulation': row['atypical_manipulation']
                    }
                    ooc_data.append(cur_data)
                    # save image
                    with open(f'{image_save_dir}/{image_id}.jpg', 'wb') as f:
                        f.write(response_image.content)
                    print(f'Crawled images: {len(ooc_data)}')

                    # save ooc_data as json
                    with open('/scratch/jin7/datasets/AMMeBa/ooc_ann_part1.json', 'w') as json_file:
                        json.dump(ooc_data, json_file, indent=4)
                    
                    all_sources.append(row['misinfo_source'])
                    all_imids.append(image_id)

print(f'Total crawled data: {len(ooc_data)}')
print(f'Total crawled images: {len(os.listdir(image_save_dir))}')
print('Done!')