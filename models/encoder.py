# Step 1: Import necessary libraries

"""
Description:
The first step is to import essential libraries for building the CNN encoder and preprocessing images:

- `torch` and `torch.nn`: Core PyTorch libraries for tensor operations and neural network layers.
- `torchvision.models`: Provides pre-trained models, including ResNet-50.
- `torchvision.transforms`: Utilities for image preprocessing, such as resizing and normalization.
- `PIL.Image`: Library for loading and manipulating images.
"""

import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import transforms
from PIL import Image

# Step 2: Define the CNN Encoder class

"""
Description:
The second step is to define the `CNNEncoder` class, which is responsible for extracting feature vectors from input images using a pre-trained ResNet-50 model. Key details include:

- **Initialization (`__init__`)**:
    - Loads the ResNet-50 model with ImageNet weights.
    - Replaces the final classification layer (`fc`) with an identity layer to output feature vectors instead of class probabilities.

- **Forward Method (`forward`)**:
    - Takes a batch of images as input.
    - Passes the images through the modified ResNet-50 model to extract feature vectors.

This class is a critical component of the image captioning pipeline, as it converts raw image data into a compact representation suitable for further processing by the decoder.
"""

class CNNEncoder(nn.Module):    
    def __init__(self):
        super(CNNEncoder, self).__init__()

        model = resnet50(weights = ResNet50_Weights.IMAGENET1K_V2)
        model.fc = nn.Identity()
        self.model = model
    
    def forward(self, images):
        features = self.model(images)
        return features
    
# Step 3: Image Preprocessing and Image -> Tensor Conversion

"""
Description:
The `preprocess_img` function prepares an input image for the CNN encoder by performing the following steps:

1. **Load Image**: Opens the image from the specified file path using the PIL library.
2. **Define Normalization Parameters**: Uses ImageNet mean and standard deviation values for normalization.
3. **Apply Transformations**:
    - Converts the image to a PyTorch tensor.
    - Normalizes the tensor using the specified mean and standard deviation.
4. **Add Batch Dimension**: Unsqueezes the tensor to add a batch dimension, making it compatible with the CNN encoder.

This function ensures that the input image is in the correct format and scale for feature extraction by the encoder.
"""

def preprocess_img(image_path: str):

    img = Image.open(image_path)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean = mean, std = std)
    ])

    tensor = transform(img)
    tensor = tensor.unsqueeze(0)
    return tensor

# Step 4: Main block for testing the CNN encoder

"""
Description:
The main block of the script is used to test the functionality of the `CNNEncoder` and `preprocess_img` function. It performs the following steps:

1. **Define Image Path**: Specifies the path to the input image to be processed.
2. **Preprocess Image**: Calls the `preprocess_img` function to prepare the image for the encoder.
3. **Initialize Encoder**: Creates an instance of the `CNNEncoder` class and sets it to evaluation mode.
4. **Extract Features**:
    - Disables gradient computation using `torch.no_grad()` to improve performance.
    - Passes the preprocessed image through the encoder to extract feature vectors.
5. **Display Results**:
    - Prints the shape of the input image tensor.
    - Prints the shape of the extracted feature vector.
    - Displays statistics (min, max, mean) of the feature vector.

This block demonstrates how to integrate the encoder and preprocessing pipeline for feature extraction in an image captioning system.
"""

if __name__ == "__main__":

    IMAGE_PATH = "../Dataset/Processed_Images/667626_18933d713e.jpg"
    processed_img = preprocess_img(IMAGE_PATH)
    print(f"Input image shape: {processed_img.shape}")

    encoder = CNNEncoder()
    encoder.eval()
    with torch.no_grad():
        features = encoder(processed_img)

    print(f"Output feature shape: {features.shape}")
    print(f"Feature stats: min = {features.min():.3f}, max = {features.max():.3f}, mean = {features.mean():.3f}")





