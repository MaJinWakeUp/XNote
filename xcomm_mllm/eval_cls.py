import json

# Paths to files
gt_path = 'D:/XComm/XCommnunityNote/final_dataset.json'
pred_path = './xcomm_mllm/results/VILA-15B_search.json'

# Load ground truth
with open(gt_path, 'r', encoding='utf-8') as f:
    gt_data = json.load(f)

# Load predictions
with open(pred_path, 'r', encoding='utf-8') as f:
    pred_data = json.load(f)

# Build gt dict: id -> label ("Yes" or "No")
gt_dict = {}
for item in gt_data:
    # Assuming each item is a dict with 'id' and 'label' keys
    # Adjust keys if necessary
    id_ = item['id']
    # label = item['label']  # "Yes" or "No"
    note = item['community_note']['summary']
    gt_dict[str(id_)] = note

# Only evaluate ids present in gt
common_ids = set(gt_dict.keys()) & set(pred_data.keys())

# Count predicted positives (predicted as "Yes")
pred_positives = [id_ for id_ in common_ids if "yes" in pred_data[id_].strip().lower()]


# Recall calculation
recall = len(pred_positives) / len(common_ids)

print(f"Total samples: {len(common_ids)}")
print(f"Recall: {recall:.4f}")