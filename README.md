# Image Captioning System

An end-to-end image captioning project built with PyTorch. It uses a ResNet-50 encoder to extract image features and an LSTM decoder to generate captions from those features.

## Features

- Caption preprocessing, cleaning, tokenization, and vocabulary creation
- CNN encoder built on top of a pretrained ResNet-50
- LSTM-based caption decoder with image feature projection
- Dataset class and collate function for batching training samples
- Small integration scripts to verify the encoder-decoder pipeline

## Project Structure

- `caption-processing.py` — cleans captions and builds vocabulary artifacts
- `models/encoder.py` — CNN encoder and image preprocessing
- `models/decoder.py` — LSTM decoder and vocabulary loading
- `training/dataset.py` — PyTorch dataset and collate function
- `training/train_config.py` — training hyperparameters, loss, and optimizer helpers
- `encoder-decoder-integration.py` — end-to-end smoke test for the pipeline
- `notebooks/Image_Captioning_Training.ipynb` — notebook for experimentation
- `Project-Learnings-and-Research/` — notes, research, and phase documentation
- `Future-Improvements/` — next-step ideas and model improvement plan

## Requirements

- Python 3.10+
- PyTorch
- torchvision
- Pillow

## Setup

1. Create and activate a virtual environment.
2. Install the required packages:

```bash
pip install torch torchvision pillow
```

## Data Preparation

The scripts expect a `Dataset/` folder containing:

- `captions.txt`
- `Processed_Images/`

Run the caption processing script to generate cleaned captions, encoded captions, and vocabulary files:

```bash
python caption-processing.py
```

## Usage

### Test the encoder

```bash
python models/encoder.py
```

### Test the decoder

```bash
python models/decoder.py
```

### Test the data pipeline

```bash
python training/dataset.py
```

### Test the full encoder-decoder flow

```bash
python encoder-decoder-integration.py
```

## Notes

- The repo currently focuses on the CNN-LSTM baseline.
- Research notes and future work are documented in the dedicated markdown folders.
- The provided scripts are intended for experimentation and pipeline verification.