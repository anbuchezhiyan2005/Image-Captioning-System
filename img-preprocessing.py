
"""
This script preprocesses images from a specified input directory by resizing them to a target size.
The processed images are then saved to an output directory.
Steps performed:
1. Creates the output directory if it does not exist.
2. Iterates through all image files (with .jpg, .jpeg, .png extensions) in the input directory.
3. For each image:
    - Reads the image using OpenCV.
    - Resizes the image to 224x224 pixels.
    - Saves the processed image to the output directory.
    - If an image cannot be read, prints an error message.
Directories:
- Input: "Dataset/Images"
- Output: "Dataset/Processed_Images"
Dependencies:
- cv2 (OpenCV)
- os
"""

import cv2
import os

original_img_dir = "Dataset/Images"
preprocessed_img_dir = "Dataset/Processed_Images"

os.makedirs(preprocessed_img_dir, exist_ok=True)

target_size = (224, 224)

for img_file in os.listdir(original_img_dir):
    if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
        img_path = os.path.join(original_img_dir, img_file)
        img = cv2.imread(img_path)

        if img is not None:
            img_resized = cv2.resize(img, target_size)
            preprocessed_path = os.path.join(preprocessed_img_dir, img_file)
            cv2.imwrite(preprocessed_path, img_resized.astype('uint8'))

        else:
            print(f"Error reading image: {img_file}")
