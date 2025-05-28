import torch
from data_parser import NewsCliping, XCommunity
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
# from google.cloud import vision
from PIL import Image
# from reverse_image_search import GoogleReverseImageSearch
import requests
from transformers import AutoModelForCausalLM

from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
from deepseek_vl2.utils.io import load_pil_images


# request = GoogleReverseImageSearch()

# def reverse_search(text, image_url):
#     response = request.response(
#         query=text,
#         image_url=image_url,
#         max_results=5,
#     )
#     return response


"""Deepseek VL2"""
# specify the path to the model
model_path = "deepseek-ai/deepseek-vl2-small"
vl_chat_processor: DeepseekVLV2Processor = DeepseekVLV2Processor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer

vl_gpt: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

PROMPT = """Follow these steps to make a logical reasoning:
            1. Summarize the claim or intent of the image-caption pair.
            2. If the claim or intent is contradicted by the image, identify the contradictions, focusing on location, time, entities, actions, and other relevant information in the image. If no, ignore this step.
            3. If the claim is contradicted by common sense or external knowledge, identify those contradictions as well. If no, ignore this step.
    """

def deepseek_inference_single(processor, model, image, caption):
    conversation = [
        {
            "role": "<|User|>",
            "content": f"Given the image: <image> \n \
                        and caption: <{caption}> \n \
                        {PROMPT}",
            # <|ref|>The giraffe at the back.<|/ref|>.",
            "images": [image],
        },
        {"role": "<|Assistant|>", "content": ""},

    ]

    # load images and prepare for inputs
    pil_images = load_pil_images(conversation)
    prepare_inputs = processor(
        conversations=conversation,
        images=pil_images,
        force_batchify=True,
        system_prompt="",
    ).to(model.device)

    # run image encoder to get the image embeddings
    inputs_embeds = model.prepare_inputs_embeds(**prepare_inputs)

    # run the model to get the response
    outputs = model.language.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=prepare_inputs.attention_mask,
        pad_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=128,
        do_sample=True,
        use_cache=False,
    )

    answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
    # print(f"{prepare_inputs['sft_format'][0]}", answer)

    return answer
""""""

# data = XCommunity("ooc_community_notes.json")
data = NewsCliping("/scratch/jin7/datasets/news_clippings/", split="val")
for i in range(0, len(data)):
    cur_data = data[i]

    # image_url, caption, label, summary = cur_data
    image_url, caption, label = cur_data
    image = Image.open(image_url)
    # image = Image.open(requests.get(image_url, stream=True).raw)
    # output = llava_inference_single(processor, model, image, caption)
    output = deepseek_inference_single(vl_chat_processor, vl_gpt, image_url, caption)

    print("========================================")
    print(f"Label: {label}")
    # print(f"Community note: {summary}")
    print(f"Caption: {caption}")
    print(f"Output: {output}")

    # detect web annotations
    # image_url = "https://pbs.twimg.com/media/F5vp9tWXQAAi-og.jpg"
    # search_results = reverse_search("demo", image_url)
    # print(search_results)
    print("========================================")
    # close the prvevious image, show the new image
    plt.close()
    plt.imshow(image)
    plt.show()



