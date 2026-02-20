# Phase 1: Data Preprocessing - Research & Learnings

## Overview
This phase focused on preparing the Flickr8k dataset for training, including image preprocessing, caption tokenization, vocabulary building, and data serialization.

---

## Research Topics

### 1. Dataset Structure Understanding
**Questions:**
- How is the Flickr8k dataset organized?
- What file formats are used for images and captions?
- How many images and captions per image?

**Key Learnings:**
- Flickr8k contains 8,000 images
- Each image has 5 different captions
- Captions are stored in text files, images in JPG format

---

### 2. Tokenization Methods
**Questions:**
- What is tokenization and why is it needed?
- What are different tokenization strategies (word-level, subword, character)?
- Which tokenization method is best for image captioning?

**Key Learnings:**
- Tokenization converts text into smaller units (tokens)
- Word-level tokenization chosen for simplicity
- Converts sentences like "A dog runs" → ["a", "dog", "runs"]

---

### 3. Vocabulary Building
**Questions:**
- What is a vocabulary in NLP?
- How do you handle rare words?
- What are special tokens and why are they needed?

**Key Learnings:**
- Vocabulary is a set of all unique words in the dataset
- Special tokens: `<pad>`, `<start>`, `<end>`, `<unk>`
- Rare words (frequency < threshold) mapped to `<unk>`

---

### 4. Image Preprocessing
**Questions:**
- What image preprocessing is needed for CNNs?
- What resizing dimensions should be used?
- Why normalize images?

**Key Learnings:**
- Resize images to consistent dimensions (e.g., 224×224 for ResNet)
- Normalize using ImageNet mean/std for transfer learning
- Convert images to tensors for PyTorch

---

## Completed Tasks
✅ Loaded Flickr8k dataset  
✅ Tokenized captions  
✅ Built vocabulary with special tokens  
✅ Processed and saved data  

---

## Output Files Generated
- `Dataset/vocabulary.json` - Complete vocabulary list
- `Dataset/Processed_Images/` - Preprocessed images
- Caption mappings and tokenized data
