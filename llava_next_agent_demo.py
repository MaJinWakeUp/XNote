from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
from data_parser import NewsCliping, XCommunity
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")

model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf", torch_dtype=torch.float16, low_cpu_mem_usage=True)
model.to("cuda:0")

def inference_single(processor, model, image, caption):
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": f"Here is the tweet paired with this image:\n <{caption}>. \n Suppose you were a 50 year old man with critical thinking, do you believe this tweet? Answer yes or no."},
            ],
        },
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(image, prompt, return_tensors="pt").to("cuda:0")

    # autoregressively complete prompt
    output = model.generate(**inputs, max_new_tokens=150)

    return processor.decode(output[0], skip_special_tokens=True)

all_labels = []
all_preds = []
# xcommu_data = XCommunity("ooc_community_notes.json")
# for i in range(70, len(xcommu_data)):
#     data = xcommu_data[i]
#     try:
#         image, caption, label, summary = data
#         output = inference_single(processor, model, image, caption)

#         print(f"Label: {label}")
#         print(f"Community note: {summary}")
#         print(f"Caption: {caption}")
#         print(f"Output: {output.split('[/INST]')[1]}")

#         # close the prvevious image, show the new image
#         plt.close()
#         plt.imshow(image)
#         plt.show()
#     except Exception as e:
#         print(e)
#         continue

all_labels = []
all_preds = []
newsclip_data = NewsCliping("/scratch/jin7/datasets/news_clippings/", split="val")
print("Number of data:", len(newsclip_data))
for data in tqdm(newsclip_data, desc="Processing news clippings"):
    image, caption, label = data
    output = inference_single(processor, model, image, caption)
    output = output.split("[/INST]")[1]
    output = output.lower()

    all_labels.append(label)
    if "yes" in output:
        all_preds.append(1)
    elif "no" in output:
        all_preds.append(0)
    else:
        all_preds.append(-1)


#  create the table as follows:
# | --- | Real | Falsified |
# | Yes |  x   |     x     |
# | No  |  x   |     x     |
# | N/A |  x   |     x     |
# Initialize the counts
results = {
    "Real": {"Yes": 0, "No": 0, "N/A": 0},
    "Falsified": {"Yes": 0, "No": 0, "N/A": 0},
}

# Populate the counts
for label, pred in zip(all_labels, all_preds):
    if label == 0:  # Real
        if pred == 1:
            results["Real"]["Yes"] += 1
        elif pred == 0:
            results["Real"]["No"] += 1
        else:
            results["Real"]["N/A"] += 1
    else:  # Falsified
        if pred == 1:
            results["Falsified"]["Yes"] += 1
        elif pred == 0:
            results["Falsified"]["No"] += 1
        else:
            results["Falsified"]["N/A"] += 1

# Create the DataFrame
df = pd.DataFrame(results)
print(df)

