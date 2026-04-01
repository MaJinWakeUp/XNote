import argparse
import base64
import io
import json
import os

from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm

from src.prompts import (
    SYS_PROMPT_DIRECT_DETECT,
    USER_PROMPT_POST_ONLY,
)


def pil_image_to_data_url(image) -> str:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def call_gpt(client, image, sys_prompt, user_prompt, model, use_web_search=False):
    image_data_url = pil_image_to_data_url(image)
    inputs = [
        {
            "role": "developer",
            "content": [{"type": "input_text", "text": sys_prompt}],
        },
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": user_prompt},
                {"type": "input_image", "image_url": image_data_url},
            ],
        },
    ]

    kwargs = {
        "model": model,
        "input": inputs,
    }
    if use_web_search:
        kwargs["tools"] = [{"type": "web_search"}]

    response = client.responses.create(**kwargs)
    return response.output_text


def main(args):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set.")

    client = OpenAI(api_key=api_key)
    dataset = load_dataset("majinwakeup30/XNote", split="test")

    current_ids = set()
    if os.path.exists(args.save_path):
        with open(args.save_path, "r") as f:
            for line in f:
                entry = json.loads(line)
                current_ids.add(entry["id"])

    total_items = len(dataset)
    processed_items = len(current_ids)
    remaining_items = total_items - processed_items

    print(f"Total items: {total_items}")
    print(f"Already processed: {processed_items}")
    print(f"Remaining: {remaining_items}")

    with open(args.save_path, "a") as f:
        for data in tqdm(dataset, desc=f"Processing {args.model_name}", total=total_items):
            item_id = data["id"]
            if item_id in current_ids:
                continue

            image = data["image"]
            text = data["text"]
            datetime = data["datetime"]

            user_prompt = USER_PROMPT_POST_ONLY.format(
                text=text,
                datetime=datetime,
            )

            try:
                response = call_gpt(
                    client=client,
                    image=image,
                    sys_prompt=SYS_PROMPT_DIRECT_DETECT,
                    user_prompt=user_prompt,
                    model=args.model_name,
                    use_web_search=args.reverse_search,
                )
                result = {"id": item_id, "response": response, "label": data["label"]}
                f.write(json.dumps(result) + "\n")
                f.flush()
            except Exception as e:
                print(f"\nError processing id {item_id}: {e}")
                result = {"id": item_id, "response": "ERROR", "label": data["label"]}
                f.write(json.dumps(result) + "\n")
                f.flush()


def parse_args():
    parser = argparse.ArgumentParser(description="Run GPT on XNote with baseline-aligned prompts.")
    parser.add_argument("--model_name", type=str, default="gpt-5", help="OpenAI model name.")
    parser.add_argument("--reverse-search", "-r", action="store_true", help="Enable OpenAI web search tool.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not args.reverse_search:
        args.save_path = os.path.join("output", f"{args.model_name}_baseline.jsonl")
    else:
        args.save_path = os.path.join("output", f"{args.model_name}_with_context.jsonl")
    os.makedirs("output", exist_ok=True)
    main(args)
