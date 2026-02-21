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

        self.projection_layer = nn.Linear(
            in_features = 2048,
            out_features = hidden_size
        )
    
    def forward(
            self, 
            word_ID_tensor: torch.Tensor, 
            image_features: torch.Tensor
    ) -> torch.Tensor:
        
        embedded_tensor = self.embedding_layer(word_ID_tensor)

        projected_features = self.projection_layer(image_features)
        initial_hidden = projected_features.unsqueeze(0)
        initial_cell = torch.zeros_like(initial_hidden)
        lstm_output, _ = self.lstm_layer(
            embedded_tensor,
            (initial_hidden, initial_cell)
        )

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

    print(f"\n Creating mock image features tensor shape: (2, 2048)")
    mock_image_features = torch.randn(2, 2048)

    # Step 10: Testing the decoder with a forward pass
    decoder.eval()
    with torch.no_grad():
        linear_output = decoder(word_id_tensor, mock_image_features)
    

    print(f"\n Output linear tensor shape: {linear_output.shape}")

    batch_size, word_sequence, vocab_output = linear_output.shape

    assert vocab_output == vocab_size, f"Output vocabulary size does not match expected vocabulary size {vocab_size}."
    assert batch_size == 2, f"Expected batch size 2, got {batch_size}"
    print(f"Decoder Testing completed! Verdict: PASS")

    # Step 11: Pipeline Summary
    print("\n" + "="*50)
    print("ENCODER-DECODER PIPELINE")
    print("="*50)
    print(f"Image Features [2, 2048]")
    print(f"    ↓ Projection")
    print(f"Projected [2, 512]")
    print(f"    ↓ Unsqueeze + Initialize LSTM")
    print(f"Initial Hidden [1, 2, 512]")
    print(f"    ↓")
    print(f"Word IDs [2, 4]")
    print(f"    ↓ Embedding")
    print(f"Embeddings [2, 4, 256]")
    print(f"    ↓ LSTM (with image context)")
    print(f"LSTM Output [2, 4, 512]")
    print(f"    ↓ Linear")
    print(f"Logits [2, 4, {vocab_size}]")
    print("="*50)
