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
    # truncate the prompt if it's too long
    # prompt = tokenizer.decode(tokenizer(prompt).input_ids[:32768 - 128])
    
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
        do_sample=True,
        # temperature=0,
        max_new_tokens=128,
    )
    text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)[0]
    text_outputs = text_outputs.strip()
    return text_outputs

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

def main():
    device = "cuda"
    device_map = "auto"
    parser = argparse.ArgumentParser(description="Run LLAVA-NEXT inference.")
    parser.add_argument("--pretrained", type=str, default="lmms-lab/llama3-llava-next-8b", help="Path to the pretrained model.") #  lmms-lab/llava-next-qwen-32b
    parser.add_argument("--model-name", type=str, default="llava_qwen", help="Name of the model.") # llava_llama3
    parser.add_argument("--reverse-search", "-r", action="store_true", help="Whether to use reverse search evidence.")
    args = parser.parse_args()

    pretrained = args.pretrained
    model_name = args.model_name
    reverse_search = args.reverse_search
    tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, device_map=device_map) #  Add any other thing you want to pass in llava_model_args

    model.eval()
    model.tie_weights()

    conv_template = "llava_llama_3" 
    # conv_template = "qwen_1_5"
    
    part = "real"  
    # Load data
    json_file = f"/scratch/jin7/datasets/XCommunityNote/{part}/final_dataset.json"
    images_dir = f"/scratch/jin7/datasets/XCommunityNote/{part}/images"
    context_dir = f"/scratch/jin7/datasets/XCommunityNote/{part}/summarized_evidence"
    with open(json_file, "r") as f:
        data = json.load(f)
    # Process each item in the data
    final_answers = {}
    k = 10
    for item in tqdm(data, desc="Processing items"):
        image_paths= item["images"]
        image_paths = [os.path.join(images_dir, img) for img in image_paths]
        if not image_paths:
            print(f"No images found for item: {item['id']}")
            final_answers[item["id"]] = ""
            continue
        
        caption = item["text"]
        date = item["date"]

        if not reverse_search:
            prompt = (
                "SYSTEM:\n"
                "You are a fact-checking assistant tasked with evaluating the accuracy of social media posts. Each post consists of an image and accompanying text. Your goal is to determine whether the claim is:\n"
                "- Misinformation: the claim is false, misleading, or taken out of context based on the image's true context.\n"
                "- Accurate: the claim accurately represents the content and context of the image.\n"
                "USER:\n"
                f"Image: <image>\n"
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
            context = get_topk_context(context, k)
            prompt = (
                "SYSTEM:\n"
                "You are a fact-checking assistant tasked with evaluating the accuracy of social media posts. Each post consists of an image and accompanying text. Your goal is to determine whether the claim is:\n"
                "- Misinformation: the claim is false, misleading, or taken out of context based on the image's true context.\n"
                "- Accurate: the claim accurately represents the content and context of the image.\n"
                "USER:\n"
                f"Image: <image>\n"
                f"Text: {caption}\n"
                f"Date: {date}\n"
                f"CONTEXT:\n{context}\n"
                "Use the provided context to assist in your evaluation. Be aware that some information in the context may also be false or misleading.\n"
                "Your response should begin with either 'Misinformation' or 'Accurate', followed by a concise explanation (1-2 sentences) referencing visual details or contextual knowledge."
            )

        # Perform inference
        try:
            response = single_inference(model, image_processor, tokenizer, prompt, image_paths[0], 
                                        conv_template=conv_template, device=device)
            response = format_response(response)
        except Exception as e:
            print(f"Error processing item {item['id']}: {e}")
            response = "Unknown. Error during processing."
            continue
        # add context to the response
        # if args.reverse_search:
        #     url = context.split("URL: ")[1].split("\n")[0].split(" ")[0] if "URL: " in context else "No URL provided"
        #     response = response + f"\n\nContext URL: {url}"
        print(f"Response: {response}")
        
        final_answers[item["id"]] = response

        # clean up memory
        torch.cuda.empty_cache()

    model_name = args.pretrained.split("/")[-1]
    # Save the final answers to a JSON file
    if reverse_search:
        output_json = f"results/{model_name}_search_{part}.json"
    else:
        output_json = f"results/{model_name}_{part}.json"
    with open(output_json, "w") as f:
        json.dump(final_answers, f, indent=4)
    
if __name__ == "__main__":
    main()