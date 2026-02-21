# Step 1: Import necessary libraries and modules
import torch
from models.encoder import CNNEncoder, preprocess_img
from models.decoder import LSTMDecoder, load_vocabulary
from pathlib import Path

# Step 2: Define the project root and image path for testing
project_root = Path(__file__).parent
IMAGE_PATH = project_root / "Dataset" / "Processed_Images" / "667626_18933d713e.jpg"
VOCABULARY_PATH = project_root / "Dataset" / "vocabulary.json"

# Step 3: Preprocess the image
print(f"\nloading the image from: {IMAGE_PATH}")
processed_img = preprocess_img(IMAGE_PATH)
print(f"processed image shape: {processed_img.shape}")

# Step 4: Load the vocabulary
vocab_size = load_vocabulary(VOCABULARY_PATH)
print(f"Vocabulary size: {vocab_size}")

# Step 5: Initialize the CNN encoder and extract image features
print("\n Extracting image features using CNN Encoder...")
encoder = CNNEncoder()
encoder.eval()
with torch.no_grad():
    image_features = encoder(processed_img)
print(f"Image features shape: {image_features.shape}")

# step 6: Creating sample captions for the image
print("\n Creating sample captions...")
sample_captions = torch.tensor([[1, 2, 3, 4, 5]])
print(f"Sample captions shape: {sample_captions.shape}")

# Step 7: Initialize the LSTM Decoder
print("\n Initializing the LSTM Decoder...")
decoder = LSTMDecoder(
    vocab_size,
    embedding_dim = 256,
    hidden_size = 512,
    num_layers = 1,
    batch_first = True
)
print("Decoder created successfully!")

# Step 8: Test the decoder with a forward pass using the sample captions and the image features
print("\n Generating caption logits...")
decoder.eval()
with torch.no_grad():
    linear_output = decoder(sample_captions, image_features)
print(f"Output logits shape: {linear_output.shape}")

# Step 9: Validate output
batch_size, seq_len, vocab_output = linear_output.shape
assert batch_size == 1, f"Batch size mismatch! Expected 1, got {batch_size}"
assert seq_len == 5, f"Sequence length mismatch! Expected 5, got {seq_len}"
assert vocab_output == vocab_size, f"Vocab size mismatch! Expected {vocab_size}, got {vocab_output}"

print("\n" + "="*60)
print("✅ END-TO-END PIPELINE TEST PASSED!")
print("="*60)

# Step 10: Pipeline Summary
print("\nPIPELINE SUMMARY:")
print("="*60)
print(f"1. Image [1, 3, 224, 224]")
print(f"   ↓ CNN Encoder (ResNet50)")
print(f"2. Features [1, 2048]")
print(f"   ↓ Projection Layer")
print(f"3. Hidden State [1, 1, 512]")
print(f"   ↓ Initialize LSTM")
print(f"4. Captions [1, 5]")
print(f"   ↓ Embedding")
print(f"5. Embeddings [1, 5, 256]")
print(f"   ↓ LSTM (with image context)")
print(f"6. LSTM Output [1, 5, 512]")
print(f"   ↓ Linear Projection")
print(f"7. Logits [1, 5, {vocab_size}]")
print("="*60)

