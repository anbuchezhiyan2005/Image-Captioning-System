# Image Captioning System

An image captioning project built around a CNN encoder and an LSTM decoder.

## Overview

This repository contains scripts and notes for an image captioning workflow that:
- preprocesses images
- cleans and tokenizes captions
- builds a vocabulary
- encodes captions into token IDs
- defines an encoder-decoder model
- tests the training configuration

## Main Components

### Model
- `models/encoder.py` — CNN encoder using ResNet-50 and an image preprocessing helper
- `models/decoder.py` — LSTM decoder with embedding, projection, and output layers

### Data Processing
- `caption-processing.py` — cleans captions, tokenizes text, builds the vocabulary, and saves processed caption files
- `img-preprocessing.py` — resizes images to `224x224` and saves them to a processed image folder
- `training/dataset.py` — dataset class and collate function for batching images and captions

### Training and Testing
- `training/train_config.py` — training settings, loss function, and optimizer helper
- `encoder-decoder-integration.py` — simple end-to-end pipeline test for the encoder and decoder

### Notebook
- `notebooks/Image_Captioning_Training.ipynb` — training notebook

## Repository Notes

The scripts reference the following paths under `Dataset/`:
- `Images/`
- `Processed_Images/`
- `captions.txt`
- `captions_encoded.json`
- `captions_cleaned.csv`
- `vocabulary.json`
- `word_to_index.json`
- `index_to_word.json`
- `vocab_metadata.json`

## Documentation Folders

- `Project-Learnings-and-Research/` — phase-wise learning notes and research
- `Future-Improvements/` — improvement ideas and planning notes

## Project Structure

```text
.
├── Future-Improvements/
├── Project-Learnings-and-Research/
├── models/
├── notebooks/
├── training/
├── caption-processing.py
├── encoder-decoder-integration.py
├── img-inspection.py
├── img-preprocessing.py
└── README.md
```
