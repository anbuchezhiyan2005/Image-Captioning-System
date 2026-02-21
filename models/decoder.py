# Step 1: Import necessary libraries and modules
import torch
import torch.nn as nn
import json
from pathlib import Path

# Step 2: Locate vocabulary.json in the project
project_root = Path(__file__).parent.parent
VOCABULARY_PATH = project_root / "Dataset" / "vocabulary.json"

# Step 3: Loading the vocabulary 
with open(VOCABULARY_PATH, mode = "r") as file:
    vocabulary = json.load(file)

# Step 4: Creating the embedding layer for the decoder
vocab_size = len(vocabulary)
embedding = nn.Embedding(num_embeddings = vocab_size, embedding_dim = 256)
print(f"Vocabulary Size: {vocab_size}")

# Step 5: Creating a sample input tensor of Word IDs and passing it to the embedding layer
word_ID_tensor = torch.tensor([1, 2, 3, 4, 5])
embedded_tensor = embedding(word_ID_tensor)

# Step 6: Verifying the output of the embedding layer
print("\nTesting Embedding Layer:")
print(f"Input word IDs: {word_ID_tensor}")
print(f"Input word IDs shape: {word_ID_tensor.shape}")
print(f"Embedded tensor shape: {embedded_tensor.shape}")

# Step 7: Creating the LSTM layer for the decoder
lstm_layer = nn.LSTM(input_size = 256, hidden_size = 512, num_layers = 1, batch_first = True)

# Step 8: Adding batch dimension to the embedded tensor
embedded_tensor_batched = embedded_tensor.unsqueeze(0)

# Step 9: Passing the embedded tensor through the LSTM layer
lstm_output, (hidden_state, cell_state) = lstm_layer(embedded_tensor_batched)

# Step 10: Verifying the output of the LSTM layer
print("\nTesting LSTM Layer:")
print(f"Input to LSTM shape: {embedded_tensor_batched.shape}")
print(f"LSTM output shape: {lstm_output.shape}")
print(f"Hidden state shape: {hidden_state.shape}")
print(f"Cell state shape: {cell_state.shape}")

# Step 11: Creating the linear layer for the decoder
linear_layer = nn.Linear(in_features = 512, out_features = vocab_size)

# Step 12: Passing the LSTM output through the linear layer
linear_output = linear_layer(lstm_output)

# Step 13: Verifying the output of the linear layer
print("\nTesting Linear Layer:")
print(f"Input shape: {lstm_output.shape}")
print(f"Output shape: {linear_output.shape}")
