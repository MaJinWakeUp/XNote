import json
from rouge_score import rouge_scorer

# Paths to files
gt_path = 'D:/XComm/XCommnunityNote/final_dataset.json'
pred_path = './xcomm_mllm/results/LLAVA-NEXT.json'

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

score = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rouge3'], use_stemmer=True)

scores = {'rouge1': [], 'rouge2': [], 'rouge3': []}

for id_ in common_ids:
    reference = gt_dict[id_]
    prediction = pred_data[id_]
    result = score.score(reference, prediction)
    for key in scores:
        scores[key].append(result[key].fmeasure)

# Compute average scores
avg_scores = {k: sum(v) / len(v) if v else 0.0 for k, v in scores.items()}

print("Average ROUGE scores:")
for k, v in avg_scores.items():
    print(f"{k}: {v:.4f}")