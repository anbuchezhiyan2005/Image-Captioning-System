import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import json
from torch.utils.data import Dataset
import os
from models.encoder import preprocess_img

class MyDataset(Dataset):
    def __init__(self, captions_path, images_path):
        self.data = self.load_caption(captions_path)
        self.add_custom_tokens()
        self.images_path = images_path


    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        caption_item =self.data[idx]
        caption_tensor = torch.tensor(caption_item["encoded_ids"])

        image_filename = caption_item["image"]
        image_tensor = self.load_image(image_filename)

        return image_tensor, caption_tensor
    
    def load_caption(self, captions_path):
        with open(captions_path, mode = "r") as f:
            captions_encoded = json.load(f)
        
        return captions_encoded
    
    def load_image(self, image_filename):
        image_path = os.path.join(self.images_path, image_filename)
        image_tensor = preprocess_img(image_path)
        return image_tensor.squeeze(0)
        
    def add_custom_tokens(self):
        for item in self.data:
            item["tokens"].insert(0, "<start>")
            item["tokens"].append("<end>")
            item["encoded_ids"].insert(0, 1)
            item["encoded_ids"].append(2)
            item["true_length"] = len(item["encoded_ids"])
        

if __name__ == "__main__":
    SAMPLE_SIZE = 3
    captions_path = PROJECT_ROOT / "Dataset" / "captions_encoded.json"
    images_path = PROJECT_ROOT / "Dataset" / "Processed_Images"

    dataset = MyDataset(captions_path, images_path)

    print(f"Dataset size: {len(dataset)}")
    print("\nTesting first 3 samples:")
    
    for i in range(SAMPLE_SIZE):
        image, caption = dataset[i]
        print(f"\nSample {i+1}:")
        print(f"  Image shape: {image.shape}")
        print(f"  Caption shape: {caption.shape}")
        print(f"  Caption (first 10 IDs): {caption[:10].tolist()}")
        print("=" * 50)