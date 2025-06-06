import csv

input_file = 'XNoteJin.csv'
output_file = 'XNoteJin_processed.csv'

def read_manual_labels(filepath):
    labels = {}
    with open(filepath, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if 'manual_label' in row and row['manual_label']:
                labels[row.get('id', len(labels))] = row['manual_label']
    return labels

# Read labels from both files
labels_mo = read_manual_labels("XNoteMo.csv")

with open(input_file, newline='', encoding='utf-8') as infile, \
    open(output_file, 'w', newline='', encoding='utf-8') as outfile:
    reader = csv.DictReader(infile)
    # Remove LLM_label and manual_topic from fieldnames
    fieldnames = [fn for fn in reader.fieldnames if fn not in ('language', 'LLM_label', 'manual_topic', 'keep')]
    writer = csv.DictWriter(outfile, fieldnames=fieldnames)
    writer.writeheader()

    for row in reader:
       # 1. Remove if keep == 'No'
       if row.get('keep', '').strip().lower() == 'no':
          continue
       # 2. Remove if language is not 'en'
       if row.get('language', '').strip().lower() != 'en':
          continue
       # 3. Remvoe if manual_label is not "context"
       if row.get('manual_label', '').strip().lower() != 'context':
          continue
       # 4. If the corresponding manual label in labels_mo is not "" and not "context", skip the row
    #    if row.get('id') in labels_mo and labels_mo[row['id']] not in ("", "context"):
    #       continue
       # 4. Remove LLM_label and manual_topic
       row.pop('language', None)
       row.pop('LLM_label', None)
       row.pop('manual_topic', None)
       row.pop('keep', None)
       
       # 5. Process LLM_topic
       llm_topic = row.get('LLM_topic', '')
       topics = [t.strip() for t in llm_topic.split(';') if t.strip()]
       if "Misattributed or Fabricated Visuals" in topics:
          if len(topics) == 1:
             topics = ["Others"]
          else:
             topics = [t for t in topics if t != "Misattributed or Fabricated Visuals"]
       row['LLM_topic'] = ';'.join(topics)
       writer.writerow(row)