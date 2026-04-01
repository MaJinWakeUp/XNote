import json
import argparse
import os
from datasets import load_dataset
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score

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

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate generation quality on deceptive XNote samples.")
    parser.add_argument(
        "--model_name",
        type=str,
        default="vila",
        help="Name of the VL model used to produce outputs.",
    )
    parser.add_argument(
        "--use_context",
        action="store_true",
        help="Whether to evaluate outputs generated with retrieved context.",
    )
    return parser.parse_args()


def load_predictions(file_path):
    """Load predictions from either JSONL entries or JSON id-to-text mapping."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Prediction file not found: {file_path}")

    pred_data = {}
    if file_path.endswith(".jsonl"):
        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                if "id" in item and "response" in item:
                    pred_data[str(item["id"])] = str(item["response"])
    else:
        with open(file_path, "r") as f:
            raw = json.load(f)
            if isinstance(raw, dict):
                pred_data = {str(k): str(v) for k, v in raw.items()}
            elif isinstance(raw, list):
                for item in raw:
                    if isinstance(item, dict) and "id" in item and "response" in item:
                        pred_data[str(item["id"])] = str(item["response"])

    return pred_data


def extract_community_note_summary(sample):
    """Extract community note summary from XNote sample in a robust way."""
    note = sample.get("community_note", "")
    if isinstance(note, dict):
        return str(note.get("summary", "")).strip()
    if isinstance(note, str):
        return note.strip()
    return ""


def compute_metrics(file_path):
    """Compute average BLEU, ROUGE-L, METEOR on deceptive samples only."""
    dataset = load_dataset("majinwakeup30/XNote", split="test")
    pred_data = load_predictions(file_path)

    total_bleu = 0
    total_rouge_l = 0
    total_meteor = 0
    count = 0


    for entry in dataset:
        if entry.get("label") != "deceptive":
            continue

        item_id = str(entry.get("id", ""))
        if not item_id or item_id not in pred_data:
            continue

        reference = extract_community_note_summary(entry)
        candidate = pred_data[item_id].strip()

        if reference and candidate:
            reference = [reference]
            bleu = calculate_bleu(reference, candidate)
            rouge_l = calculate_rouge_l(reference, candidate)
            meteor = calculate_meteor(reference, candidate)
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
    print(f"Evaluated samples: {count}")
    print(f"Average BLEU: {avg_bleu}")
    print(f"Average ROUGE-L: {avg_rouge_l}")
    print(f"Average METEOR: {avg_meteor}")


if __name__ == "__main__":
    args = parse_args()
    postfix = "with_context" if args.use_context else "baseline"
    file_path = f"./output/{args.model_name}_{postfix}.jsonl"
    print(f"Model: {args.model_name}")
    print(f"Mode: {postfix}")
    print(f"Prediction file: {file_path}")
    compute_metrics(file_path)