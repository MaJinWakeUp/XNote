import requests
from PIL import Image
import torch
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from data_parser import AMMeBa
import matplotlib.pyplot as plt

# Load the model in half-precision
processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-vicuna-7b-hf")
model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-vicuna-7b-hf", torch_dtype=torch.float16, low_cpu_mem_usage=True)
model.to("cuda:0")

dataset = AMMeBa(img_dir="/scratch/jin7/datasets/AMMeBa/images_part1/", ann_file="/scratch/jin7/datasets/AMMeBa/demo_refine.json")

SYSTEM_PROMPT = {
    "role": "system",
    "content": [
        {"type": "text", "text": "A chatbox assistant to help user identify potential misinformation post by answering questions. \
            For each question, make sure to provide brief, concise, and confident answers without bias."},
    ],
}

def llava_run(processor, model, image, conversation):
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    # print(prompt)
    inputs = processor(image, prompt, return_tensors="pt").to("cuda:0")

    # autoregressively complete prompt
    output = model.generate(**inputs, max_new_tokens=256)
    out_text = processor.decode(output[0], skip_special_tokens=True)
    answer = out_text.split("ASSISTANT:")[-1]
    return answer

for idx in range(len(dataset)):
    data = dataset[idx]
    image_path = data["image_path"]
    caption = data["caption"]
    image = Image.open(image_path)

    # Don't need to ask Q1 anymore
    # conversation1 = [
    #     SYSTEM_PROMPT,
    #     {
    #         "role": "user",
    #         "content": [
    #             {"type": "image"},
    #             {"type": "text", "text": f"Q1: Is there any text in this image? If yes, what does it say?"},
    #         ],
    #     },
    # ]
    
    # answer1 = llava_run(processor, model, image, conversation1)

    conversation1a = [
        SYSTEM_PROMPT,
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": f"Q1a: Analyze the image and its caption: {caption}.  \
                What is the claim of this post? Describe with neccessary details such as time, location, entity, event etc.."},
            ],
        },
    ]
    answer1a = llava_run(processor, model, image, conversation1a)

    conversation1b = conversation1a + [
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": answer1a},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Q1b: Now analyze what is the implicit intent or motivation of this post?"},
            ],
        },
    ]
    answer1b = llava_run(processor, model, image, conversation1b)

    conversation2 = conversation1b + [
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": answer1b},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Q2: Are the claim and intent of this post inconsistent with the image itself, or contrary to common facts and expert knowledge?"},
            ],
        },
    ]
    answer2 = llava_run(processor, model, image, conversation2)

    # conversation2b = conversation2a + [
    #     {
    #         "role": "assistant",
    #         "content": [
    #             {"type": "text", "text": answer2a},
    #         ],
    #     },
    #     {
    #         "role": "user",
    #         "content": [
    #             {"type": "text", "text": "Q2b: Is there any inconsistency between the claim in Q1a and common facts, or between the intent in Q1b and common facts?"},
    #         ],
    #     },
    # ]
    # answer2b = llava_run(processor, model, image, conversation2b)

    # show the image and the generated text
    print("========================================")
    # print(f"Image path: {image_path}")
    print(f"Caption: {caption}")
    print(f"Q1a: {answer1a}")
    print(f"Q1b: {answer1b}")
    print(f"Q2a: {answer2}")
    print("========================================")
    plt.imshow(image)
    plt.show()

"""
# Get three different images
url = "https://www.ilankelman.org/stopsigns/australia.jpg"
image_stop = Image.open(requests.get(url, stream=True).raw)

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image_cats = Image.open(requests.get(url, stream=True).raw)

url = "https://huggingface.co/microsoft/kosmos-2-patch14-224/resolve/main/snowman.jpg"
image_snowman = Image.open(requests.get(url, stream=True).raw)

# Prepare a batch of two prompts, where the first one is a multi-turn conversation and the second is not
conversation_1 = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "What is shown in this image?"},
            ],
    },
    {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "There is a red stop sign in the image."},
            ],
    },
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "What about this image? How many cats do you see?"},
            ],
    },
]

conversation_2 = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "What is shown in this image?"},
            ],
    },
]

prompt_1 = processor.apply_chat_template(conversation_1, add_generation_prompt=True)
prompt_2 = processor.apply_chat_template(conversation_2, add_generation_prompt=True)
prompts = [prompt_1, prompt_2]

# We can simply feed images in the order they have to be used in the text prompt
# Each "<image>" token uses one image leaving the next for the subsequent "<image>" tokens
inputs = processor(images=[image_stop, image_cats, image_snowman], text=prompts, padding=True, return_tensors="pt").to(model.device)

# Generate
generate_ids = model.generate(**inputs, max_new_tokens=30)
processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)"
"""