import json
import time
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import pandas as pd


# crawl time for demo.json
# all_data = json.load(open('/scratch/jin7/datasets/AMMeBa/demo.json', 'r'))
# print(len(all_data))

"""
data format:
    {
        "misinfo_source": "https://archive.ph/faaYG",
        "image_id": "45a08df91bb91a7def4ab6f0dc9edd9fa37a102ba2e3de835aa6a229",
        "caption": "WOW! Facebook analytics reveal a clean sweep for Donald Trump in this election! Sweeping 42 of the 50 states! INCLUDING NEW YORK!",
        "platform": "Others",
        "CheckTime": "2023.03.23 16:20:56",
        "self_miscontextualizing": "true",
        "circumstance_manipulation": "true",
        "identity_manipulation": "false",
        "location_manipulation": "false",
        "datetime_manipulation": "false",
        "atypical_manipulation": false
    },
"""
# headers = {
#     "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:93.0) Gecko/20100101 Firefox/93.0",
# }
# add_pubtime = []
# for data in tqdm(all_data, desc="Processing"):
#     url = data['misinfo_source']
#     res = requests.get(url, headers=headers)
#     time.sleep(1)
#     soup = BeautifulSoup(res.content, 'html.parser')
#     possible_times = []
#     # 1. <meta property="article:published_time">
#     meta_pubtime = soup.find('meta', property='article:published_time')
#     if meta_pubtime:
#         possible_times.append(meta_pubtime['content'])
#     # 2. <span class="timestampContent">
#     span_pubtime = soup.find('span', class_='timestampContent')
#     if span_pubtime:
#         possible_times.append(span_pubtime.text)
#     # 3. <time>
#     time_pubtime = soup.find('time')
#     if time_pubtime:
#         possible_times.append(time_pubtime['datetime'])
#     # 4. <span class="time">
#     span_time = soup.find('span', class_='time')
#     if span_time:
#         possible_times.append(span_time.text)
    
#     # use the first possible time
#     print(possible_times)
#     if possible_times:
#         data['PubTime'] = possible_times[0]
#     else:
#         data['PubTime'] = None
#     add_pubtime.append(data)

# json.dump(add_pubtime, open('/scratch/jin7/datasets/AMMeBa/demo_.json', 'w'), indent=4)

stage2_file = '/scratch/jin7/datasets/AMMeBa/stage_2.csv'
stage2_df = pd.read_csv(stage2_file)
# read the demo_.json, for each data, find the corresponding row with same misinfo_source in stage2_df, then add the  'image_watermark', 'image_annotation', 'image_text', 'image_text_misinfo_relevance', 'reverse_image_search' to the data
demo_data = json.load(open('/scratch/jin7/datasets/AMMeBa/demo_refine.json', 'r'))
for data in tqdm(demo_data, desc="Processing"):
    misinfo_source = data['misinfo_source']
    row = stage2_df[stage2_df['misinfo_source'] == misinfo_source]
    if row.shape[0] > 0:
        row = row.iloc[0]
        # data['image_watermark'] = str(row['image_watermark']).lower() if pd.notna(row['image_watermark']) else None
        # data['image_annotation'] = str(row['image_annotation']).lower() if pd.notna(row['image_annotation']) else None
        # data['image_text'] = str(row['image_text']).lower() if pd.notna(row['image_text']) else None
        # data['image_text_misinfo_relevance'] = str(row['image_text_misinfo_relevance']).lower() if pd.notna(row['image_text_misinfo_relevance']) else None
        # data['reverse_image_search'] = str(row['reverse_image_search']).lower() if pd.notna(row['reverse_image_search']) else None
        data['fact_check_url'] = str(row['fact_check_url']).lower() if pd.notna(row['fact_check_url']) else None
    else:
        # data['image_watermark'] = None
        # data['image_annotation'] = None
        # data['image_text'] = None
        # data['image_text_misinfo_relevance'] = None
        # data['reverse_image_search'] = None
        data['fact_check_url'] = None
json.dump(demo_data, open('/scratch/jin7/datasets/AMMeBa/demo_refine.json', 'w'), indent=4)