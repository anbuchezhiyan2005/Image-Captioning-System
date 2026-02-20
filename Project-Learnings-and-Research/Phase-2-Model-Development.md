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
*(To be filled after research)*

---

## Progress Tracker

### Completed
✅ Task 1: CNN Encoder with ResNet50  
✅ Mini-Task 2A: Research Word Embeddings  
✅ Mini-Task 2B: Implement Embedding Layer  

### In Progress
⏳ Mini-Task 2C: Research LSTM Basics  
⏳ Mini-Task 2D: Implement LSTM Layer  

### Todo
📋 Mini-Task 2E: Add Output Projection Layer  
📋 Mini-Task 2F: Integrate Image Features with Decoder  
📋 Task 3: Combine Encoder-Decoder  
📋 Task 4: Implement Training Loop  

---

## Code Files Created
- `models/encoder.py` - CNN Encoder (ResNet50 feature extractor)
- `models/decoder.py` - LSTM Decoder (in progress)

---

## Resources Used
- PyTorch Documentation: https://pytorch.org/docs/stable/
- Torchvision Models: https://pytorch.org/vision/stable/models.html
- Research papers on image captioning architectures
