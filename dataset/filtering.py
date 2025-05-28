import os
import json
from langdetect import detect

ori_json = "/scratch/jin7/datasets/XCommnunityNote/dataset_4-30.json"

# Read the original JSON file
with open(ori_json, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Process each element in the list
for element in data:
    if "text" in element:
        try:
            element["language"] = detect(element["text"])
        except Exception as e:
            element["language"] = "unknown"

# Save the updated data to a new file
new_json = "/scratch/jin7/datasets/XCommnunityNote/dataset_with_language.json"
with open(new_json, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

# Filter the data based on the detected language, only keep English and unknown
filtered_data = [element for element in data if element["language"] == "en" or element["language"] == "unknown"]
# Save the filtered data to a new file
filtered_json = "/scratch/jin7/datasets/XCommnunityNote/dataset_filtered.json"
with open(filtered_json, 'w', encoding='utf-8') as f:
    json.dump(filtered_data, f, ensure_ascii=False, indent=4)
# Check the number of elements in the filtered data
print(f"Number of elements in the filtered data: {len(filtered_data)}")