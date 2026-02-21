# Step 1: Import necessary libraries and modules
import torch
import torch.nn as nn
import json
from pathlib import Path

# Step 2: Define the LSTMDecoder class
class LSTMDecoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_size: int,
        num_layers: int,
        batch_first: bool
    ):
        super(LSTMDecoder, self).__init__()
        self.embedding_layer = nn.Embedding(
            num_embeddings = vocab_size,
            embedding_dim = embedding_dim
        )

        self.lstm_layer = nn.LSTM(
            input_size = embedding_dim,
            hidden_size = hidden_size,
            num_layers = num_layers,
            batch_first = batch_first
        )

        self.linear_layer = nn.Linear(
            in_features = hidden_size,
            out_features = vocab_size
        )
    
    def forward(self, word_ID_tensor: torch.Tensor) -> torch.Tensor:
        embedded_tensor = self.embedding_layer(word_ID_tensor)
        lstm_output, _ = self.lstm_layer(embedded_tensor)
        linear_output = self.linear_layer(lstm_output)

        return linear_output
    
# Step 3: Define a function to load the vocabulary and return its size
def load_vocabulary(vocabulary_path: Path) -> int:
    with open(vocabulary_path, mode = "r") as file:
        vocabulary = json.load(file)
    
    return len(vocabulary)
    
# Step 4: Main block to test the decoder implementation
if __name__ == "__main__":

    # Step 5: Define the project root and paths to necessary files
    project_root = Path(__file__).parent.parent

    # Step 6: Define hyperparameters for the decoder
    VOCABULARY_PATH = project_root / "Dataset" / "vocabulary.json"
    EMBEDDING_DIM = 256
    HIDDEN_SIZE = 512
    NUM_LAYERS = 1

    # Step 7: Loading the vocabulary 
    vocab_size = load_vocabulary(VOCABULARY_PATH)
    print(f"Vocabulary size: {vocab_size}")

    # Step 8: Creating a decoder instance
    decoder = LSTMDecoder(
        vocab_size = vocab_size,
        embedding_dim = EMBEDDING_DIM,
        hidden_size = HIDDEN_SIZE,
        num_layers = NUM_LAYERS,
        batch_first = True,
    )
    print(f"\nDecoder architecture:\n{decoder}")

    print("\n" + "="*50)
    print("TESTING DECODER")
    print("=" * 50)

    # Step 9: Creating a sample input tensor for the decoder
    word_id_tensor = torch.tensor(
        [
            [1, 2, 3, 4],
            [5, 6, 7, 8]
        ]
    )

    print(f"\n Input word ID tensor shape: {word_id_tensor.shape}")

    # Step 10: Testing the decoder with a forward pass
    decoder.eval()
    with torch.no_grad():
        linear_output = decoder(word_id_tensor)
    

    print(f"\n Output linear tensor shape: {linear_output.shape}")

    batch_size, word_sequence, vocab_output = linear_output.shape

    assert vocab_output == vocab_size, f"Output vocabulary size does not match expected vocabulary size {vocab_size}."
    print(f"Decoder Testing completed! Verdict: PASS")