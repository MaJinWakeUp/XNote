import json
import argparse
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate fact-checking model predictions on test set."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="vila",
        help="Name of the VL model to evaluate (default: vila)",
    )
    parser.add_argument(
        "--use_context",
        action="store_true",
        help="Whether context was used in predictions (enables with_context mode).",
    )
    return parser.parse_args()


def main(args):
    # Determine result file suffix based on context usage
    postfix = "with_context" if args.use_context else "baseline"
    result_path = f"output/{args.model_name}_{postfix}.jsonl"

    results = []
    # Load predictions
    with open(result_path, 'r') as f:
        for line in f:
            item = json.loads(line)
            results.append(item)

    gts = []
    preds = []

    for item in results:
        pred_label = None
        response = item['response'].lower()
        if response.startswith("false"):
            pred_label = 1
        elif response.startswith("true"):
            pred_label = 0
        else:
            pred_label = 1  
        
        gt_label = item['label']
        if gt_label == "deceptive":
            gt = 1
        else:
            gt = 0
        preds.append(pred_label)
        gts.append(gt)

    assert len(gts) == len(preds), "Ground truth and predictions length mismatch"

    # Calculate metrics
    def compute_fp_fn(gts, preds):
        false_positives = sum(1 for gt, pred in zip(gts, preds) if gt == 0 and pred == 1)
        false_negatives = sum(1 for gt, pred in zip(gts, preds) if gt == 1 and pred == 0)
        return false_positives, false_negatives

    false_positives, false_negatives = compute_fp_fn(gts, preds)
    accuracy = accuracy_score(gts, preds)
    precision = precision_score(gts, preds)
    recall = recall_score(gts, preds)
    f1 = f1_score(gts, preds)

    # Print results
    print(f"Model: {args.model_name}")
    print(f"Mode: {'with_context' if args.use_context else 'baseline'}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"False Positives: {false_positives}")
    print(f"False Negatives: {false_negatives}")


if __name__ == "__main__":
    args = parse_args()
    main(args)