import torch
import torch.nn as nn
import json
from pathlib import Path
VOCABULARY_PATH = "../Dataset/vocabulary.json"

project_root = Path(__file__).parent.parent
VOCABULARY_PATH = project_root / "Dataset" / "Vocabulary.json"

with open(VOCABULARY_PATH, mode = "r") as file:
    vocab_data = json.load(file)

vocab_size = len(vocab_data)
embedding = nn.Embedding(num_embeddings = vocab_size, embedding_dim = 256)
print(f"Vocabulary Size: {vocab_size}")