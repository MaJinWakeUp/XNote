import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

model = "InternVL3-14B"  # Specify the model name
search = False  # Set to True if using reverse search evidence
MAC = False

misinfo_gt_path = f"/scratch/jin7/datasets/XCommunityNote/misinformation/final_dataset.json"
real_gt_path = f"/scratch/jin7/datasets/XCommunityNote/real/final_dataset.json"
# if search:
#     misinfo_pred_path = f"./results/{model}_search_misinformation.json"
#     real_pred_path = f"./results/{model}_search_real.json"
# else:
#     misinfo_pred_path = f"./results/{model}_misinformation_enhance.json"
#     real_pred_path = f"./results/{model}_real_enhance.json"

if not search:
    misinfo_pred_path = f"./results/{model}_misinformation.json"
    real_pred_path = f"./results/{model}_real.json"
elif not MAC:
    misinfo_pred_path = f"./results/{model}_misinformation_search.json"
    real_pred_path = f"./results/{model}_real_search.json"
else:
    misinfo_pred_path = f"./results/{model}_misinformation_search_MAC.json"
    real_pred_path = f"./results/{model}_real_search_MAC.json"

# Load ground truth
# with open(misinfo_gt_path, 'r', encoding='utf-8') as f:
#     gt_data = json.load(f)

# Load predictions
with open(misinfo_pred_path, 'r', encoding='utf-8') as f:
    misinfo_pred_data = json.load(f)

with open(real_pred_path, 'r', encoding='utf-8') as f:
    real_pred_data = json.load(f)

gts = []
preds = []

positive_labels = ['misinformation', 'misleading', 'false'] #"yes"
negative_labels = ['accurate', 'true'] # "no"

for item in misinfo_pred_data:
    cur_label = None
    for pl in positive_labels:
        if pl in misinfo_pred_data[item].lower():
            cur_label = 1
            break
    if cur_label is None:
        for nl in negative_labels:
            if nl in misinfo_pred_data[item].lower():
                cur_label = 0
                break
    if cur_label is not None:
        preds.append(cur_label)
    else:
        preds.append(1)
    gts.append(1)  # All items in misinfo_gt are labeled as 1 (misinformation)

for item in real_pred_data:
    cur_label = None
    for pl in positive_labels:
        if pl in real_pred_data[item].lower():
            cur_label = 1
            break
    if cur_label is None:
        for nl in negative_labels:
            if nl in real_pred_data[item].lower():
                cur_label = 0
                break
    if cur_label is not None:
        preds.append(cur_label)
    else:
        preds.append(1)
    gts.append(0)  # All items in real_gt are labeled as 0 (real news)

assert len(gts) == len(preds), "Ground truth and predictions length mismatch"
# Calculate metrics
def compute_fp_fn(gts, preds):
    false_positives = sum(1 for gt, pred in zip(gts, preds) if gt == 0 and pred == 1)
    false_negatives = sum(1 for gt, pred in zip(gts, preds) if gt == 1 and pred == 0)
    return false_positives, false_negatives

false_positives, false_negatives = compute_fp_fn(gts, preds)
print(f"False Positives: {false_positives}")
print(f"False Negatives: {false_negatives}")
accuracy = accuracy_score(gts, preds)
precision = precision_score(gts, preds)
recall = recall_score(gts, preds)
f1 = f1_score(gts, preds)

# Print results
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")