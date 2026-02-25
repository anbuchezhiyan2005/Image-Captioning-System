
import csv
import re
import string
from collections import Counter
import json
import os
from typing import TypedDict

class VocabArtifacts(TypedDict):
    vocabulary: list[str]
    word_frequency: dict[str, int]
    word_to_index: dict[str, int]
    index_to_word: dict[int, str]
    vocab_size: int

captions_file = "Dataset/captions.txt"
vocabulary_file = "Dataset/vocabulary.json"

PUNCTUATION_TRANSLATOR = str.maketrans("", "", string.punctuation)

def load_captions(captions_path: str) -> list[dict[str, str]]:
    
    rows = []

    with open(captions_path, mode = 'r', encoding = "utf-8", newline = "") as file:
        reader = csv.DictReader(file)

        expected_cols = {"image", "caption"}
        if not expected_cols.issubset(set(reader.fieldnames or [])):
            raise ValueError(
                f"Invalid header in {captions_path}. Expected columns: image, caption. "
            )

        for row in reader:
            image = (row.get("image") or "").strip()
            raw_caption = (row.get("caption") or "").strip()

            if not image or not raw_caption:
                continue

            clean_caption = clean_text(raw_caption)
            rows.append({"image": image, "caption": clean_caption})
    
    return rows


def clean_text(text: str) -> str:
    
    if not isinstance(text, str):
        return ""
    
    text = text.lower()
    text = text.translate(PUNCTUATION_TRANSLATOR)
    text = re.sub(r"\s+", " ", text) # Replace multiple whitespace with a single space. For example, "This   is  a   test." becomes "This is a test."
    text = text.strip()
    
    return text


def tokenize_text(cleaned_text: str) -> list[str]:

    if not isinstance(cleaned_text, str):
        return []
    
    if not cleaned_text:
        return []
    
    return cleaned_text.split(" ")

def tokenize_captions(rows: list[dict[str, str]]) -> list[dict[str, object]]:

    tokenized_rows: list[dict[str, object]] = []

    for row in rows:
        image = row["image"]
        caption = row["caption"]
        tokens = tokenize_text(caption)

        tokenized_rows.append({
            "image": image,
            "caption": caption,
            "tokens": tokens  
        })
    
    return tokenized_rows



def build_vocabulary(
        tokenized_rows: list[dict[str, object]],
        min_freq: int = 1,
        special_tokens: list[str] | None = None
    ) -> VocabArtifacts:
    
    if special_tokens is None:
        special_tokens = ["<pad>", "<start>", "<end>", "<unk>"]
    
    word_frequency = Counter()

    for row in tokenized_rows:
        tokens = row.get("tokens", [])
        
        if isinstance(tokens, list):
            word_frequency.update(tokens)
    
    filtered_words = [w for w, c in word_frequency.items() if c >= min_freq]
    filtered_words = sorted(filtered_words)
    filtered_words = [w for w in filtered_words if w not in set(special_tokens)]

    vocabulary = special_tokens + filtered_words

    word_to_index = {word: idx for idx, word in enumerate(vocabulary)}
    index_to_word = {idx: word for word, idx in word_to_index.items()}

    return {
        "vocabulary": vocabulary,
        "word_frequency": dict(word_frequency),
        "word_to_index": word_to_index,
        "index_to_word": index_to_word,
        "vocab_size": len(vocabulary)
    }

def encode_single_caption(
    tokens: list[str],
    word_to_index: dict[str, int],
    start_token: str = "<start>",
    end_token: str = "<end>",
    unk_token: str = "<unk>"
    ) -> tuple[list[int], int]:
    
    start_id = word_to_index[start_token]
    end_id = word_to_index[end_token]
    unk_id = word_to_index[unk_token]

    token_ids = [word_to_index.get(token, unk_id) for token in tokens]

    seq = [start_id] + token_ids + [end_id]

    true_length = len(seq)
    
    return seq, true_length


def encode_captions(
        tokenized_rows: list[dict[str, object]],
        word_to_index: dict[str, int],
    ) -> list[dict[str, object]]:

    encoded_rows: list[dict[str, object]] = []
    
    for row in tokenized_rows:
        image = row["image"]
        caption = row["caption"]
        tokens = row.get("tokens", [])

        if not isinstance(tokens, list):
            tokens = []
        
        encoded_ids, true_length = encode_single_caption(
            tokens = tokens,
            word_to_index = word_to_index,
        )

        encoded_rows.append(
            {
            "image": image,
            "caption": caption,
            "tokens": tokens,
            "encoded_ids": encoded_ids,
            "true_length": true_length
            }
        )

    return encoded_rows


def save_json(data: object, filepath: str):
    parent_dir = os.path.dirname(filepath)
    if parent_dir:
      os.makedirs(parent_dir, exist_ok = True)  

    with open(filepath, mode = "w", encoding = "utf-8") as file:
        json.dump(data, file, indent = 2)
    
    print(f"Saved in location: {filepath}")
        

def save_vocabulary(
        vocab_artifacts: VocabArtifacts, 
        output_dir: str = "Dataset"
    ):
    
    os.makedirs(output_dir, exist_ok = True)

    vocab_list_path = os.path.join(output_dir, "vocabulary.json")
    save_json(vocab_artifacts["vocabulary"], vocab_list_path)

    word_to_index_path = os.path.join(output_dir, "word_to_index.json")
    save_json(vocab_artifacts["word_to_index"], word_to_index_path)

    index_to_word_path = os.path.join(output_dir, "index_to_word.json")
    index_to_word_serializable = {str(k): v for k, v in vocab_artifacts["index_to_word"].items()}
    save_json(index_to_word_serializable, index_to_word_path)

    metadata = {
        "vocab_size": vocab_artifacts["vocab_size"],
        "word_frequency": vocab_artifacts["word_frequency"],
        "special_tokens": ["<pad>", "<start>", "<end>", "<unk>"]
    }

    metadata_path = os.path.join(output_dir, "vocab_metadata.json")
    save_json(metadata, metadata_path)

    print(f"Vocabulary artifacts saved to {output_dir}")

def save_encoded_captions(
        encoded_rows: list[dict[str, object]],
        output_dir: str = "Dataset"
    ):

    os.makedirs(output_dir, exist_ok = True)

    output_path = os.path.join(output_dir, "captions_encoded.json")
    save_json(encoded_rows, output_path)

    print(f"Encoded captions saved to {output_path}")

def save_cleaned_captions_csv(
        tokenized_rows: list[dict[str, object]],
        output_dir = "Dataset"    
    ):

    os.makedirs(output_dir, exist_ok = True)
    output_path = os.path.join(output_dir, "captions_cleaned.csv")

    with open(output_path, mode = "w", encoding = "utf-8", newline = "") as file:
        writer = csv.DictWriter(file, fieldnames = ["image", "caption", "tokens"])
        writer.writeheader()

        for row in tokenized_rows:
            tokens = row.get("tokens", [])

            if isinstance(tokens, list):
                tokens_str = " ".join(tokens)
            else:
                tokens_str = ""
            
            writer.writerow({
                "image": row["image"],
                "caption": row["caption"],
                "tokens": tokens_str
            })
        
    print(f"Cleaned captions saved to {output_path}")


if __name__ == "__main__":

    captions_rows = load_captions(captions_file)
    print(f"Loaded {len(captions_rows)} captions from {captions_file}")

    tokenized_rows = tokenize_captions(captions_rows)
    print(f"Tokenized captions {len(tokenized_rows)} rows")

    vocab_artifacts = build_vocabulary(tokenized_rows, min_freq = 1)
    print(f"Vocabulary size: {vocab_artifacts['vocab_size']} unique words")

    encoded_rows = encode_captions(
        tokenized_rows = tokenized_rows,
        word_to_index = vocab_artifacts["word_to_index"],
    )

    save_vocabulary(vocab_artifacts, output_dir = "Dataset")
    save_encoded_captions(encoded_rows, output_dir = "Dataset")
    save_cleaned_captions_csv(tokenized_rows, output_dir = "Dataset")

    print("Caption processing completed successfully.")
    print("Phase 1: Data Preparation Complete!")




        
