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

def get_schema_from_python_path(path: str) -> str:
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

    # Load data
    json_file = "/scratch/jin7/datasets/XCommnunityNote/dataset_filtered.json"
    images_dir = "/scratch/jin7/datasets/XCommnunityNote/images"
    search_dir = "/scratch/jin7/datasets/XCommnunityNote/summarized_evidence"
    with open(json_file, "r") as f:
        data = json.load(f)
    # Process each item in the data
    final_answers = {}
    for item in tqdm(data, desc="Processing items"):
        image_urls = item["image_urls"]
        image_paths = []
        for image_url in image_urls:
            # Check if the image URL is a local path
            if os.path.isfile(image_url):
                image_paths.append(image_url)
            else:
                # If not, download the image from the URL
                image_name = os.path.basename(image_url)
                image_path = os.path.join(images_dir, image_name)
                if os.path.exists(image_path):
                    image_paths.append(image_path)
                else:
                    # print(f"Image not found: {image_path}")
                    continue
        # If no images were found, skip this item
        if not image_paths:
            print(f"No images found for item: {item['id']}")
            final_answers[item["id"]] = ""
            continue
        
        captions = item["text"]

        if not args.reverse_search:
            prompt = f"Here is a tweet with the image and text: {captions}. \
                Is this tweet misinformation? Please provide a brief explanation. \
                Your answer should start with a 'Yes', 'No', or 'Not sure'. "
        else:
            # construct new prompt with reverse search evidence
            cur_evidence_file = os.path.join(search_dir, f"{item['id']}.txt")
            if not os.path.exists(cur_evidence_file):
                evidence = ""
                print(f"Evidence file not found for item: {item['id']}")
            else:
                # read evidence from the file
                with open(cur_evidence_file, "r") as f:
                    evidence = f.read()
            prompt = f"Here is a tweet with the image and text: {captions}. \
                Here are some related evidences: {evidence}. \n \
                According to the evidence, is this tweet misinformation? Please provide a brief explanation. \
                Your answer should start with a 'Yes', 'No', or 'Not sure'. "

        # Perform inference
        response = single_inference(model, prompt, image_paths, response_format=response_format)
        print(f"Response: {response}")
        final_answers[item["id"]] = response

    # Save the final answers to a JSON file
    if args.reverse_search:
        output_json = "results/VILA-15B_search.json"
    else:
        output_json = "results/VILA-15B.json"
    with open(output_json, "w") as f:
        json.dump(final_answers, f, indent=4)

if __name__ == "__main__":
    main()
