
"""
This script inspects a batch of image files in a specified directory, printing basic information for each image.
- Scans the 'Dataset/Images' directory for image files with .jpg, .jpeg, or .png extensions.
- Limits the inspection to a batch size of 20 images.
- For each image, prints:
    - Index and filename
    - Dimensions (width x height)
    - Number of channels
    - File size in bytes
- Handles errors gracefully, reporting issues with loading or processing images.
Dependencies:
    - cv2 (OpenCV)
    - os
"""
import cv2
import os

img_directory = "Dataset/Images"

img_batch_size = 20

img_files = [f for f in os.listdir(img_directory) if f.endswith(('.jpg', '.jpeg', '.png'))]
print(f"Total images found: {len(img_files)}")
img_files = img_files[:img_batch_size]

for idx, img_file in enumerate(img_files, start = 1):
    img_path = os.path.join(img_directory, img_file)

    try:
        img = cv2.imread(img_path)
        if img is None:
            print(f"Error: Unable to load {img_file}")
            continue
        height, width, channels = img.shape
        file_size = os.path.getsize(img_path)

        print(f"Image {idx}: {img_file}")
        print(f" - Dimensions: {width}x{height}")
        print(f" - Channels: {channels}")
        print(f" - File Size: {file_size} bytes")
        print("-" * 30)
    
    except Exception as e:
        print(f"Error processing {img_file}: {e}")
