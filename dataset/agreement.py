import csv

def read_manual_labels(filepath):
    labels = {}
    with open(filepath, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if 'manual_label' in row and row['manual_label']:
                labels[row.get('id', len(labels))] = row['manual_label']
    return labels

# Read labels from both files
labels_jin = read_manual_labels("XNoteJin.csv")
labels_mo = read_manual_labels("XNoteMo.csv")

# Find common keys
common_keys = set(labels_jin.keys()) & set(labels_mo.keys())

# Calculate agreement
agree = 0
total = 0
for key in common_keys:
    if labels_jin[key] == "context" or labels_mo[key] == "context":
        total += 1
        if labels_jin[key] == "context" and labels_mo[key] == "context":
            agree += 1

agreement_score = agree / total if total > 0 else 0

print(f"Agreement score: {agreement_score:.4f} ({agree}/{total})")