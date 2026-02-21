# Phase 2: Model Development - Research & Learnings

## Overview
This phase focuses on building the encoder-decoder architecture for image captioning using CNN (ResNet50) and LSTM.

---

## Task 1: CNN Encoder (ResNet50)

### Research Topic: Transfer Learning with ResNet50
**Questions:**
- What is transfer learning?
- Why use pre-trained models?
- What is ResNet50 and how does it work?
- What is the final layer of ResNet50 and why remove it?

**Key Learnings:**
- Transfer learning reuses weights from models trained on large datasets (ImageNet)
- ResNet50 has 50 layers and extracts rich image features
- Final classification layer (fc) outputs 1000 class probabilities
- For feature extraction, replace fc layer with Identity layer
- Output: 2048-dimensional feature vector per image

---

### Research Topic: Image Preprocessing for Pre-trained Models
**Questions:**
- What preprocessing is required for pre-trained models?
- What are ImageNet normalization values?
- Why add a batch dimension?

**Key Learnings:**
- Pre-trained models expect specific input preprocessing
- ImageNet normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
- Batch dimension needed even for single image: [1, 3, H, W]
- Use `.eval()` mode and `torch.no_grad()` for inference

---

## Task 2: LSTM Decoder

### Research Topic: Word Embeddings
**Date:** February 20, 2026

**Questions:**
1. What does `nn.Embedding` do?
2. What are the two main parameters: `num_embeddings` and `embedding_dim`?
3. How does it convert word IDs (integers) to dense vectors?
4. Why use embeddings instead of one-hot encoding?

**Key Learnings:**
- `nn.Embedding` is a lookup table that converts word IDs to dense vectors
- `num_embeddings`: Size of vocabulary (how many words)
- `embedding_dim`: Dimensionality of each word's vector representation (e.g., 256)
- Word ID (integer) → Dense vector [0.2, -0.5, 0.8, ..., 0.1] (learning representation)
- Embeddings are more efficient than one-hot encoding:
  - One-hot: [0, 0, 0, ..., 1, ..., 0] (sparse, vocab_size dimensions)
  - Embedding: [0.2, -0.5, 0.8, ...] (dense, embedding_dim dimensions)
- Embeddings capture semantic relationships between words

**Embedding Dimension Trade-offs:**
- Small (64-128): Fast but less expressive
- Medium (256): Good balance for medium datasets like Flickr8k ✅
- Large (512-1024): Rich representations but slower, for large datasets

---

### Research Topic: LSTM (Long Short-Term Memory)
**Date:** February 20, 2026

**Questions to Research:**
1. What does `nn.LSTM` do?
   - What kind of data does it process?
   - What does it output?

2. What are the key parameters?
   - `input_size`: What is this?
   - `hidden_size`: What is this?
   - `num_layers`: What does this control?
   - `batch_first`: What does `True` vs `False` mean?

3. What does LSTM return?
   - What is the `output`?
   - What is the `hidden state`?
   - What's the difference between them?

4. Input/Output shapes:
   - If input is `[batch, seq_len, input_size]`, what's the output shape?

**Key Learnings:**
- **Purpose**: LSTM processes sequential data and captures temporal dependencies (order matters)
- **Key Parameters**:
  - `input_size`: Dimension of each input element (e.g., 256 for embedding dimension)
  - `hidden_size`: Dimension of LSTM's internal hidden state (e.g., 512)
  - `num_layers`: Number of stacked LSTM layers (depth)
  - `batch_first=True`: Changes input shape from [seq_len, batch, features] to [batch, seq_len, features]
- **Outputs**:
  - `output`: Contains hidden states for ALL time steps → Shape: [batch, seq_len, hidden_size]
  - `hidden`: Contains only the LAST time step's hidden state → Shape: [num_layers, batch, hidden_size]
  - Use `output` when you need predictions at every time step
  - Use `hidden` when you only need the final state (or to initialize next sequence)
- **Shape Transformation Example**:
  - Input: [32, 10, 256] (32 batches, 10 words, 256-dim embeddings)
  - Output: [32, 10, 512] (32 batches, 10 time steps, 512-dim hidden states)

**Quiz Score:** 4/4 Perfect! ✅

**Common Pitfalls & Misconceptions:**
1. **Confusion: `cell_state` vs `lstm_output`**
   - ❌ Mistake: Thinking `cell_state` is the final output to use for predictions
   - ✅ Reality: 
     - `lstm_output` contains hidden states for ALL time steps → Use this for predictions
     - `hidden_state` is only the LAST time step's hidden state → Use for continuing generation
     - `cell_state` is internal memory at last time step → Use for continuing generation
   - **Rule**: For caption generation, always use `lstm_output` and pass it through the linear layer

2. **Understanding the relationship**:
   - `lstm_output[:, -1, :]` (last element) == `hidden_state[-1, :, :]` (for single-layer LSTM)
   - The last position of `lstm_output` IS the `hidden_state`

3. **Shape confusion**:
   - `lstm_output`: [batch, seq_len, hidden_size] - one hidden state per word
   - `hidden_state`: [num_layers, batch, hidden_size] - only the final hidden state
   - `cell_state`: [num_layers, batch, hidden_size] - only the final cell state

---

### Research Topic: Linear Layers (nn.Linear)
**Date:** February 21, 2026

**Questions to Research:**
1. What does `nn.Linear` do?
   - What is its purpose in a neural network?
   - What kind of data does it process?

2. What are the key parameters?
   - `in_features`: What does this represent?
   - `out_features`: What does this represent?

3. How does it work?
   - What happens when you pass data through a linear layer?
   - What is the shape of the input and output?

4. What mathematical operation does it perform?

**Key Learnings:**
- **Purpose**: Applies a linear transformation (affine transformation) to the input data
- **Mathematical Operation**: `y = x @ W^T + b`
  - Matrix multiplication with weight matrix `W`
  - Addition of bias vector `b`
- **Key Parameters**:
  - `in_features`: Dimensionality of input features (e.g., 512 from LSTM hidden size)
  - `out_features`: Dimensionality of output features (e.g., 8256 for vocab size)
- **Shape Transformation Rule**: Linear layers transform **ONLY the last dimension**
  - Input: `[batch, seq_len, in_features]` → Output: `[batch, seq_len, out_features]`
  - All dimensions except the last are preserved
  - Example: `[1, 5, 512]` through `nn.Linear(512, 8256)` → `[1, 5, 8256]`
- **Use Case in Decoder**: 
  - Maps LSTM hidden states to vocabulary logits
  - Each position in the sequence gets a probability distribution over all words

**Quiz Score:** 3/4 - Good! ✅

**Common Pitfalls & Misconceptions:**
1. **Shape transformation mistake**:
   - ❌ Mistake: Thinking `nn.Linear(512, 8256)` on input `[1, 5, 512]` produces `[1, 8256]`
   - ✅ Reality: Output is `[1, 5, 8256]` - linear layer transforms ONLY the last dimension
   - **Rule**: All dimensions except the last are preserved (batch size, sequence length stay the same)

2. **Understanding dimensionality**:
   - The linear layer applies the same transformation to each element along the preserved dimensions
   - For `[batch, seq_len, features]`, it processes each of the `seq_len` positions independently
   - Think of it as applying the linear transformation `seq_len` times (once per word)

---

### Decoder Refactoring Experience
**Date:** February 21, 2026

**Task**: Refactor procedural decoder code into reusable `LSTMDecoder` class.

**Common Mistakes During Refactoring:**

1. **Storing input data in constructor**:
   - ❌ Mistake: Passing `word_ID_tensor` to `__init__` and storing it as instance variable
   - ✅ Reality: Constructor should only define **architecture**, not data
   - **Rule**: Data should be passed to `forward()` method, not stored in the model

2. **Wrong linear layer input size**:
   - ❌ Mistake: Using `embedding_dim` as `in_features` for linear layer
   - ✅ Reality: Linear layer receives LSTM output, so use `hidden_size`
   - **Rule**: Check what the previous layer outputs before defining the next layer's input size

3. **Incorrect LSTM unpacking**:
   - ❌ Mistake: `lstm_output = self.lstm_layer(...)` (captures tuple, not tensor)
   - ✅ Reality: `lstm_output, _ = self.lstm_layer(...)` (unpack to get actual output)
   - **Rule**: LSTM returns `(output, (hidden, cell))` - always unpack properly

4. **Trying to unpack Linear layer output**:
   - ❌ Mistake: `linear_output, _ = self.linear_layer(...)` (Linear returns single tensor)
   - ✅ Reality: `linear_output = self.linear_layer(...)` (no unpacking needed)
   - **Rule**: Only LSTM/GRU return tuples; most other layers return single tensors

5. **Manual batch dimension handling**:
   - ❌ Mistake: Adding `.unsqueeze(0)` inside forward when input already has batch dimension
   - ✅ Reality: Input is already `[batch, seq_len]`, embedding produces `[batch, seq_len, embed_dim]`
   - **Rule**: Understand input shape before manually adding dimensions

**Key Lesson**: 
- Model classes define **architecture** (layers and their connections)
- Data flows through `forward()` method
- Always verify tensor shapes at each step

---

## Progress Tracker

### Completed
✅ Task 1: CNN Encoder with ResNet50  
✅ Task 2: LSTM Decoder (Complete!) 🎉
  - ✅ Mini-Task 2A: Research Word Embeddings  
  - ✅ Mini-Task 2B: Implement Embedding Layer  
  - ✅ Mini-Task 2C: Research LSTM Basics  
  - ✅ Mini-Task 2D: Implement LSTM Layer
  - ✅ Mini-Task 2E: Research & Implement Linear Layer
  - ✅ Mini-Task 2F: Refactor into LSTMDecoder Class
  - ✅ Mini-Task 2G: Test & Verify All Components

### In Progress
⏳ None - Ready for Task 3!

### Todo  
📋 Task 3: Combine Encoder-Decoder
  - Integrate image features with decoder
  - Initialize LSTM hidden state with image features
  - Test full pipeline (image → caption logits)
📋 Task 4: Loss Function & Optimizer
📋 Task 5: Implement Training Loop  

---

## Code Files Created
- `models/encoder.py` - CNN Encoder (ResNet50 feature extractor) ✅ Complete
- `models/decoder.py` - LSTM Decoder (LSTMDecoder class) ✅ Complete

## Session Summary
**Date:** February 20-21, 2026

**Total Learning Time:** ~4-5 hours  
**Tasks Completed:** Encoder (Task 1) + Decoder (Task 2)  
**Overall Phase 2 Progress:** ~65% complete

---

## Resources Used
- PyTorch Documentation: https://pytorch.org/docs/stable/
- Torchvision Models: https://pytorch.org/vision/stable/models.html
- Research papers on image captioning architectures
