import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import json
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
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
            if item["encoded_ids"][0] != 1:
                item["tokens"].insert(0, "<start>")
                item["encoded_ids"].insert(0, 1)

            if item["encoded_ids"][-1] != 0 and item["encoded_ids"][-1] != 2:
                item["tokens"].append("<end>")
                item["encoded_ids"].append(2)

            item["true_length"] = len(item["encoded_ids"])
        

def collate_fn(batch):
    images = []
    captions = []

    for img, caption in batch:
        images.append(img)
        captions.append(caption)
    
    batched_images = torch.stack(images)
    batched_captions = pad_sequence(
        captions,
        batch_first = True,
        padding_value = 0 
    )

    return batched_images, batched_captions

if __name__ == "__main__":
    SAMPLE_SIZE = 3
    BATCH_SIZE = 8

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
    
    print("\n" + "=" * 50)
    print("TESTING DATALOADER WITH COLLATE FUNCTION")
    print("=" * 50)

    dataloader = DataLoader(
        dataset,
        batch_size = BATCH_SIZE,
        shuffle = False,
        collate_fn = collate_fn
    )

    batch_images, batch_captions = next(iter(dataloader))

    print(f"\nBatch size: {BATCH_SIZE}")
    print(f"Batched images shape: {batch_images.shape}")
    print(f"Batched captions shape: {batch_captions.shape}")
    print(f"\nFirst caption in batch:\n{batch_captions[0]}")
    print(f"\nLast caption in batch:\n{batch_captions[-1]}")

    print("\n" + "="*60)
    print("VERIFYING PADDING")
    print("="*60)
    for i in range(BATCH_SIZE):
        caption = batch_captions[i]
        num_pads = (caption == 0).sum()
        print(f"Caption {i+1}: Length={len(caption)}, Padding tokens={num_pads}")
    
    print("\n✅ DATALOADER TEST PASSED!")