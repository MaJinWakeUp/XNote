import os
import json
from PIL import Image
from torch.utils.data import Dataset
import requests

class NewsCliping(Dataset):
    def __init__(self, data_dir, split='val'):
        self.data_dir = data_dir
        self.split = split
        self.data = self.load_data()
    
    def load_data(self):
        visual_news_data_ann = os.path.join(self.data_dir, "visual_news/origin/data.json")
        visual_news_data = json.load(open(visual_news_data_ann))
        visual_news_data_map = {ann["id"]: ann for ann in visual_news_data}

        data_ann = os.path.join(self.data_dir, f"news_clippings/data/merged_balanced/{self.split}.json")
        data = json.load(open(data_ann))
        annotations = data["annotations"]
        
        all_data = []
        for ann in annotations:
            caption = visual_news_data_map[ann["id"]]["caption"]
            image_path = visual_news_data_map[ann["image_id"]]["image_path"]
            falsified = ann["falsified"]
            all_data.append({"image_path": image_path, "caption": caption, "falsified": falsified})
        
        return all_data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = os.path.join(self.data_dir, "visual_news/origin", item["image_path"])
        # print(image_path)
        # image = Image.open(image_path)
        caption = item["caption"]
        falsified = item["falsified"]
        
        return image_path, caption, falsified

class XCommunity(Dataset):
    def __init__(self, data_file):
        self.data_file = data_file
        self.data = self.load_data()
    
    def load_data(self):
        data = json.load(open(self.data_file))
        all_data = []
        for item in data:
            image_path = item["image_urls"][0]
            caption = item["text"]
            label = item["community_note"]["classification"]
            summary = item["community_note"]["summary"]
            all_data.append({"image_path": image_path, "caption": caption, "label": label, "summary": summary})
        
        return all_data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = item["image_path"]
        # image = Image.open(requests.get(image_path, stream=True).raw)
        caption = item["caption"]
        label = item["label"]
        summary = item["summary"]
        
        return image_path, caption, label, summary

class AMMeBa(Dataset):
    def __init__(self, img_dir, ann_file):
        self.img_dir = img_dir
        self.ann_file = ann_file
        self.data = json.load(open(self.ann_file))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        meta = self.data[idx]
        image_id = meta["image_id"]
        image_path = os.path.join(self.img_dir, image_id + ".jpg")
        meta["image_path"] = image_path
        return meta