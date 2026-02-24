import torch
from pathlib import Path
import json

class Dataset():
    def __init__(self, captions_encoded):
        self.captions_encoded = captions_encoded

    def add_custom_tokens(self):
        for item in self.captions_encoded:
            item["tokens"].insert(0, "<start>")
            item["tokens"].append("<end>")
            item["encoded_ids"].insert(0, 1)
            item["encoded_ids"].append(2)
            item["true_length"] = len(item["encoded_ids"])
        
        return self.captions_encoded

    def convert_captions_to_tensors(self):
        tensors = []

        for item in self.captions_encoded:
            tensors.append(torch.tensor(item["encoded_ids"]))
        
        return tensors

if __name__ == "__main__":
    SAMPLE_SIZE = 5

    PROJECT_ROOT = Path(__file__).parent.parent
    captions_path = PROJECT_ROOT / "Dataset" / "captions_encoded.json"
    with open(captions_path, "r") as f:
        captions_encoded = json.load(f)

    dataset = Dataset(captions_encoded[:SAMPLE_SIZE])
    captions_with_tokens = dataset.add_custom_tokens()

    print("\nSample captions with custom tokens:")
    for item in captions_with_tokens:
        print(f"Caption: {item['caption']}")
        print(f"Tokens: {item['tokens']}\n")
    
    print("\nConverted tensors:")
    tensors = dataset.convert_captions_to_tensors()
    for i, tensor in enumerate(tensors, start = 1):
        print(f"Tensor {i}: {tensor}")