from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle

from PIL import Image
import requests
import copy
import torch
import os 
import json
from tqdm import tqdm
import argparse

def single_inference(model, image_processor, tokenizer, prompt, image_path, conv_template="llava_llama_3", device="cuda"):
    # Load the image
    image = Image.open(image_path).convert("RGB")
    image_tensor = process_images([image], image_processor, model.config)
    image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]
    question = DEFAULT_IMAGE_TOKEN + prompt
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.tokenizer = tokenizer
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
    image_sizes = [image.size]

    cont = model.generate(
        input_ids,
        images=image_tensor,
        pad_token_id=tokenizer.eos_token_id,
        image_sizes=image_sizes,
        do_sample=False,
        temperature=0,
        max_new_tokens=128,
    )
    text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)[0]
    text_outputs = text_outputs.strip()
    return text_outputs

def main():
    device = "cuda"
    device_map = "auto"
    parser = argparse.ArgumentParser(description="Run LLAVA-NEXT inference.")
    parser.add_argument("--pretrained", type=str, default="lmms-lab/llama3-llava-next-8b", help="Path to the pretrained model.")
    parser.add_argument("--model-name", type=str, default="llava_llama3", help="Name of the model.")
    parser.add_argument("--reverse-search", "-r", action="store_true", help="Whether to use reverse search evidence.")
    args = parser.parse_args()

    pretrained = args.pretrained
    model_name = args.model_name
    reverse_search = args.reverse_search
    tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, device_map=device_map) # Add any other thing you want to pass in llava_model_args

    model.eval()
    model.tie_weights()

    conv_template = "llava_llama_3" # Make sure you use correct chat template for different models
    
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

        if not reverse_search:
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
        response = single_inference(model, image_processor, tokenizer, prompt, image_paths[0], conv_template=conv_template, device=device)
        print(f"Response: {response}")
        final_answers[item["id"]] = response

    # Save the final answers to a JSON file
    if reverse_search:
        output_json = "results/LLAVA-NEXT_search.json"
    else:
        output_json = "results/LLAVA-NEXT.json"
    with open(output_json, "w") as f:
        json.dump(final_answers, f, indent=4)
    
if __name__ == "__main__":
    main()