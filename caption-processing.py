
"""
This script processes image captions to build a vocabulary of unique words.
The script performs the following steps:
1. Reads image captions from a text file ('Dataset/captions.txt'), where each line contains an identifier and a caption separated by a comma.
2. Normalizes the captions by:
    - Converting all text to lowercase.
    - Removing punctuation.
3. Splits the captions into individual words and accumulates them into a set to ensure uniqueness.
4. Saves the resulting vocabulary as a JSON file ('Dataset/vocabulary.json').
5. Outputs the total number of unique words in the vocabulary.
Modules:
- string: Used for handling and removing punctuation.
- json: Used for saving the vocabulary as a JSON file.
Variables:
- captions_file: Path to the input file containing image captions.
- vocabulary_file: Path to the output file where the vocabulary will be saved.
- vocabulary: A set that stores unique words from the captions.
Output:
- Prints the total number of unique words in the vocabulary.
- Saves the vocabulary as a JSON file.

"""

import string
import json

captions_file = "Dataset/captions.txt"
vocabulary_file = "Dataset/vocabulary.json"

vocabulary = set()

with open(captions_file, 'r') as file:
    for line in file:
        _, caption = line.strip().split(',', 1)
        caption = caption.lower().translate(str.maketrans('', '', string.punctuation))

        words = caption.split()
        vocabulary.update(words)

with open(vocabulary_file, 'w') as vocab_file:
    json.dump(list(vocabulary), vocab_file, indent=2)

print(f"Total unique words in vocabulary: {len(vocabulary)}")
print(f"Vocabulary saved to {vocabulary_file}")
        
