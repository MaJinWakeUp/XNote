import csv
import json
import os

csv_file = 'XNoteJin_processed.csv'
json_file = 'dataset_filtered.json'
output_file = 'final_dataset.json'
reverse_search_directory = 'reverse_search/'
if not os.path.exists(reverse_search_directory):
    os.makedirs(reverse_search_directory)

# Load JSON data as a dict keyed by id for fast lookup
with open(json_file, 'r', encoding='utf-8') as f:
    json_data = json.load(f)
    json_dict = {str(item['id']): item for item in json_data}

final_data = []
images_list = os.listdir('images/')

with open(csv_file, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        row_id = str(row['id'])
        if row_id in json_dict:
            data = json_dict[row_id].copy()
            # Add manual_label and LLM_topic from CSV
            data['manual_label'] = row.get('manual_label', '')
            # Convert "LLM_topic" to a list
            llm_topic = row.get('LLM_topic', '')
            if llm_topic:
                data['LLM_topic'] = [topic.strip() for topic in llm_topic.split(';') if topic.strip()]
            else:
                data['LLM_topic'] = []
            # Convert image_urls to images with basename
            image_urls = data.get('image_urls', [])
            data['images'] = [os.path.basename(url) for url in image_urls]
            # Remove original image_urls
            if 'image_urls' in data:
                del data['image_urls']
            # if the image in the list does not exist in folder images/, remove the data
            data['images'] = [img for img in data['images'] if img in images_list]
            if not data['images']:
                continue
            
            # for data['community_note'], which is a dict, only keep 
            # "classification", 
            # "misleadingOther", 
            # "misleadingFactualError",
            # "misleadingManipulatedMedia",
            # "misleadingOutdatedInformation"
            # "misleadingMissingImportantContext"
            # "misleadingUnverifiedClaimAsFact"
            # "misleadingSatire"
            if 'community_note' in data and isinstance(data['community_note'], dict):
                misleading_keys = [
                    "classification", 
                    "misleadingOther", 
                    "misleadingFactualError",
                    "misleadingManipulatedMedia",
                    "misleadingOutdatedInformation",
                    "misleadingMissingImportantContext",
                    "misleadingUnverifiedClaimAsFact",
                    "misleadingSatire",
                    "summary"
                ]
                data['community_note'] = {k: v for k, v in data['community_note'].items() if k in misleading_keys}
            # remove "llm_image_classification", "full_llm_image_response", "topical_categories", "full_topical_categories_response", "language"
            
            # save the "reverse_image_search_results" "dememe_reverse_image_search_results" and "dememe_reverse_image_text" to another json file in reverse_search_directory with name {id}.json
            search_json = {
                "reverse_image_search_results": data.get("reverse_image_search_results", []),
                "dememe_reverse_image_search_results": data.get("dememe_reverse_image_search_results", []),
                "dememe_reverse_image_text": data.get("dememe_reverse_image_text", "")
            }
            # directly save
            with open(os.path.join(reverse_search_directory, f"{row_id}.json"), 'w', encoding='utf-8') as search_file:
                json.dump(search_json, search_file, ensure_ascii=False, indent=2)
            
            keys_to_remove = [
                "llm_image_classification", 
                "full_llm_image_response", 
                "topical_categories", 
                "full_topical_categories_response", 
                "language",
                "reverse_image_search_results",
                "dememe_reverse_image_search_results",
                "dememe_reverse_image_text"
            ]
            for key in keys_to_remove:
                if key in data:
                    del data[key]

            final_data.append(data)

print(f"Processed {len(final_data)} entries from {csv_file}.")
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(final_data, f, ensure_ascii=False, indent=2)