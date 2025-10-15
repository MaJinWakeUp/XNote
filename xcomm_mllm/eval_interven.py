import sys
import json
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score
from sklearn.metrics import precision_score, recall_score, f1_score

def calculate_bleu(reference, candidate):
    """
    Calculate the average BLEU score (BLEU-1, BLEU-2, BLEU-3, BLEU-4) 
    between a reference and a candidate sentence.
    """
    reference = [ref.split() for ref in reference]
    candidate = candidate.split()
    smoothie = SmoothingFunction().method5

    bleu1 = sentence_bleu(reference, candidate, weights=(1, 0, 0, 0), smoothing_function=smoothie)
    bleu2 = sentence_bleu(reference, candidate, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothie)
    bleu3 = sentence_bleu(reference, candidate, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothie)
    bleu4 = sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie)

    return (bleu1 + bleu2 + bleu3 + bleu4) / 4

def calculate_rouge_l(reference, candidate):
    """
    Calculate ROUGE-L score between a reference and a candidate sentence.
    """
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = [scorer.score(ref, candidate)['rougeL'].fmeasure for ref in reference]
    return max(scores)

def calculate_meteor(reference, candidate):
    """
    Calculate METEOR score between a reference and a candidate sentence.
    """
    reference = [ref.split() for ref in reference]
    candidate = candidate.split()
    return meteor_score(reference, candidate)

def compute_metrics(file_path):
    """
    Compute average BLEU, ROUGE-L, and METEOR scores for a JSONL file.
    """
    data_file = "/scratch/jin7/datasets/XCommunityNote/misinformation/final_dataset.json"
    gt_data = json.load(open(data_file, "r"))

    total_bleu = 0
    total_rouge_l = 0
    total_meteor = 0
    count = 0


    pred_data = json.load(open(file_path, "r"))

    for entry in gt_data:
        id = str(entry["id"])
        if id not in pred_data:
            print(f"ID {id} not found in predictions.")
            continue
        reference = entry["community_note"]["summary"].strip()
        candidata = pred_data[id].strip()
        if reference and candidata:
            reference = [reference]
            bleu = calculate_bleu(reference, candidata)
            rouge_l = calculate_rouge_l(reference, candidata)
            meteor = calculate_meteor(reference, candidata)
            total_bleu += bleu
            total_rouge_l += rouge_l
            total_meteor += meteor
            count += 1
        
    if count == 0:
        print("No valid data found.")
        return
    
    avg_bleu = total_bleu / count
    avg_rouge_l = total_rouge_l / count
    avg_meteor = total_meteor / count
    print(f"Average BLEU: {avg_bleu}")
    print(f"Average ROUGE-L: {avg_rouge_l}")
    print(f"Average METEOR: {avg_meteor}")

if __name__ == "__main__":
    model_name = "llama3-llava-next-8b"  # Specify the model name
    search = True
    MAC = True
    if not search:
        file_path = f"./results/{model_name}_misinformation.json"
    elif not MAC:
        file_path = f"./results/{model_name}_misinformation_search.json"
    else:
        file_path = f"./results/{model_name}_misinformation_search_MAC.json"
    compute_metrics(file_path)