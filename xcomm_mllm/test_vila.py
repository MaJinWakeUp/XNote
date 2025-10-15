import os
import argparse
import importlib.util
import json

from pydantic import BaseModel
from termcolor import colored

import llava
from llava import conversation as clib
from llava.media import Image, Video
from llava.model.configuration_llava import JsonSchemaResponseFormat, ResponseFormat
# from transformers import GenerationConfig
from tqdm import tqdm
import torch

def get_schema_from_python_path(path):
    schema_path = os.path.abspath(path)
    spec = importlib.util.spec_from_file_location("schema_module", schema_path)
    schema_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(schema_module)

    # Get the Main class from the loaded module
    Main = schema_module.Main
    assert issubclass(
        Main, BaseModel
    ), f"The provided python file {path} does not contain a class Main that describes a JSON schema"
    return Main.schema_json()

def single_inference(model, prompt, media, response_format=None):
    final_prompt = []
    # Prepare multi-modal prompt
    if media is not None:
        for media_item in media or []:
            if any(media_item.endswith(ext) for ext in [".jpg", ".jpeg", ".png"]):
                media_item = Image(media_item)
            elif any(media_item.endswith(ext) for ext in [".mp4", ".mkv", ".webm"]):
                media_item = Video(media_item)
            else:
                raise ValueError(f"Unsupported media type: {media_item}")
            final_prompt.append(media_item)
    if prompt is not None:
        final_prompt.append(prompt)
    # Generate response, set the max_new_tokens
    generation_config = model.default_generation_config
    generation_config.max_new_tokens = 128
    response = model.generate_content(final_prompt, response_format=response_format,
                                      generation_config=generation_config)
    return response

def get_topk_context(context, k = 1):
    # context contains "URL: xxx\nSummary: xxx\nURL: xxx\nSummary: xxx\n..."
    if not context:
        return ""
    context_items = context.split("URL: ")
    context_items = [item.strip() for item in context_items if item.strip()]
    if len(context_items) <= k:
        return context
    # Select top k items
    selected_items = context_items[:k]
    # Format the selected items
    formatted_context = "\n".join([f"URL: {item}" for item in selected_items])
    return formatted_context

def format_response(response):
    response = response.strip().replace("\n", " ")
    return response

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", "-m", type=str, default="Efficient-Large-Model/NVILA-15B")
    parser.add_argument("--conv-mode", "-c", type=str, default="auto")
    # parser.add_argument("--text", type=str)
    # parser.add_argument("--media", type=str, nargs="+")
    parser.add_argument("--json-mode", action="store_true")
    parser.add_argument("--json-schema", type=str, default=None)
    parser.add_argument("--reverse-search", "-r", action="store_true", help="Enable reverse search")
    args = parser.parse_args()

    
    # Convert json mode to response format
    if not args.json_mode:
        response_format = None
    elif args.json_schema is None:
        response_format = ResponseFormat(type="json_object")
    else:
        schema_str = get_schema_from_python_path(args.json_schema)
        print(schema_str)
        response_format = ResponseFormat(type="json_schema", json_schema=JsonSchemaResponseFormat(schema=schema_str))

    # Load model
    model = llava.load(args.model_path)

    # Set conversation mode
    clib.default_conversation = clib.conv_templates[args.conv_mode].copy()

    part = "real"
    k = 10

    # Load data
    json_file = f"/scratch/jin7/datasets/XCommunityNote/{part}/final_dataset.json"
    images_dir = f"/scratch/jin7/datasets/XCommunityNote/{part}/images"
    context_dir = f"/scratch/jin7/datasets/XCommunityNote/{part}/summarized_evidence"

    with open(json_file, "r") as f:
        data = json.load(f)
    # Process each item in the data
    final_answers = {}
    for item in tqdm(data, desc="Processing items"):
        image_paths = item["images"]
        image_paths = [os.path.join(images_dir, img) for img in image_paths]
        if not image_paths:
            print(f"No images found for item: {item['id']}")
            final_answers[item["id"]] = ""
            continue
        
        caption = item["text"]
        date = item["date"]

        if not args.reverse_search:
            prompt = (
                "SYSTEM:\n"
                "You are a fact-checking assistant tasked with evaluating the accuracy of social media posts. Each post consists of an image and accompanying text. Your goal is to determine whether the claim is:\n"
                "- Misinformation: the claim is false, misleading, or taken out of context based on the image's true context.\n"
                "- Accurate: the claim accurately represents the content and context of the image.\n"
                "USER:\n"
                f"Image: |input image|\n"
                f"Text: {caption}\n"
                f"Date: {date}\n"
                "Your response should begin with either 'Misinformation' or 'Accurate', followed by a concise explanation (1-2 sentences) referencing visual details or contextual knowledge."
            )
        else:
            # construct new prompt with reverse search evidence
            cur_context_file = os.path.join(context_dir, f"{item['id']}.txt")
            if not os.path.exists(cur_context_file):
                context = ""
                print(f"Context file not found for item: {item['id']}")
            else:
                # read evidence from the file
                with open(cur_context_file, "r") as f:
                    context = f.read()
            # # retrieve context from the vectorstore
            context = get_topk_context(context, k=k)
            prompt = (
                "SYSTEM:\n"
                "You are a fact-checking assistant tasked with evaluating the accuracy of social media posts. Each post consists of an image and accompanying text. Your goal is to determine whether the claim is:\n"
                "- Misinformation: the claim is false, misleading, or taken out of context based on the image's true context.\n"
                "- Accurate: the claim accurately represents the content and context of the image.\n"
                "USER:\n"
                f"Image: |input image|\n"
                f"Text: {caption}\n"
                f"Date: {date}\n"
                f"CONTEXT:\n{context}\n"
                "Use the provided context to assist in your evaluation. Be aware that some information in the context may also be false or misleading.\n"
                "Your response should begin with either 'Misinformation' or 'Accurate', followed by a concise explanation (1-2 sentences) referencing visual details or contextual knowledge."
            )

        # Perform inference
        try:
            response = single_inference(model, prompt, image_paths, response_format=response_format)
            response = format_response(response)
        except Exception as e:
            print(f"Error processing item {item['id']}: {e}")
            response = "Unknown. Error during processing."
            continue
        """
        # add context to the response
        if args.reverse_search:
            # context contains "URL: xxx\nSummary: xxx\nURL: xxx\nSummary: xxx\n..."
            urls = []
            for line in context.split("\n"):
                if line.startswith("URL: "):
                    url = line[5:].strip().split()[0]
                    urls.append(url)
            urls_ = "\n".join(urls)     
            response = response + f"\nContext URL: {urls_}"
        """
        print(f"Response: {response}")
        
        final_answers[item["id"]] = response

        # clean up memory
        torch.cuda.empty_cache()

    model_name = args.model_path.split("/")[-1]
    # Save the final answers to a JSON file
    if args.reverse_search:
        output_json = f"results/{model_name}_{part}_search.json"
    else:
        output_json = f"results/{model_name}_{part}.json"
    with open(output_json, "w") as f:
        json.dump(final_answers, f, indent=4)

if __name__ == "__main__":
    main()
