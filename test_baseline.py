import os
import json
import argparse
from tqdm import tqdm
from src.agents import VLAgent
from src.prompts import (
    SYS_PROMPT_DIRECT_DETECT,
    SYS_PROMPT_WITH_CONTEXT,
    USER_PROMPT_POST_ONLY,
    USER_PROMPT_WITH_CONTEXT,
)
from datasets import load_dataset

def main(args):
    dataset = load_dataset("majinwakeup30/XNote", split="test")
    model = VLAgent(
        model_name=args.model_name,
        max_new_tokens=args.max_new_tokens
    )

    current_ids = set()
    if os.path.exists(args.save_path):
        with open(args.save_path, 'r') as f:
            for line in f:
                entry = json.loads(line)
                current_ids.add(entry['id'])

    # Calculate number of items to process
    total_items = len(dataset)
    processed_items = len(current_ids)
    remaining_items = total_items - processed_items
    
    print(f"Total items: {total_items}")
    print(f"Already processed: {processed_items}")
    print(f"Remaining: {remaining_items}")

    with open(args.save_path, 'a') as f:  # Open in append mode to skip overwriting existing data
        for data in tqdm(dataset, desc=f"Processing {args.model_name}", total=total_items):
            id = data['id']
            if id in current_ids:
                continue

            image = data['image']
            text = data['text']
            datetime = data['datetime']
            retrieved_context = data['summarized_evidence']
            if args.use_context and retrieved_context:
                sys_prompt = SYS_PROMPT_WITH_CONTEXT
                user_prompt = USER_PROMPT_WITH_CONTEXT.format(
                    text=text,
                    datetime=datetime,
                    retrieved_context=retrieved_context,
                )
            else:
                sys_prompt = SYS_PROMPT_DIRECT_DETECT
                user_prompt = USER_PROMPT_POST_ONLY.format(
                    text=text,
                    datetime=datetime,
                )
            try:
                response = model.chat(
                    input_images=image,
                    system_prompt=sys_prompt,
                    user_prompt=user_prompt,
                    do_sample=args.do_sample,
                    temperature=args.temperature,
                )
                result = {'id': id, 'response': response, 'label': data['label']}
                f.write(json.dumps(result) + '\n')
                f.flush()
            except Exception as e:
                print(f"\nError processing id {id}: {e}")
                # Optionally save error info
                result = {'id': id, 'response': "ERROR", 'label': data['label']}
                f.write(json.dumps(result) + '\n')
                f.flush()


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Test VLAgent for misinformation detection.")
    parser.add_argument("--model_name", type=str, default="vila", help="Name of the VL model to use: ['gemma', 'internvl', 'llavaonevision', 'qwen', 'vila']")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Maximum number of new tokens to generate.")
    parser.add_argument("--use_context", action='store_true', help="Whether to use context information.")
    parser.add_argument("--do_sample", action='store_true', default=True, help="Whether to use sampling during generation.")
    parser.add_argument("--temperature", type=float, default=0.3, help="Temperature for sampling during generation.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if not args.use_context:
        args.save_path = os.path.join("output", f"{args.model_name}_baseline.jsonl")
    else:
        args.save_path = os.path.join("output", f"{args.model_name}_with_context.jsonl")
    os.makedirs("output", exist_ok=True)
    main(args)