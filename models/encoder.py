# Step 1: Import necessary libraries
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
from torchvision import transforms

# Step 2: Load the pre-trained ResNet-50 model
model = resnet50(weights = ResNet50_Weights.IMAGENET1K_V2)

# Step 3: Display the model architecture (optional)
# print(model)

# Step 4: Removing the final classification layer
fc = model.fc

# Step 5: Adding a placeholder layer to replace the removed classification layer
model.fc = nn.Identity() # Ignore type error

# Step 6: Displaying the final classification layer after replacement:
# print(model.fc)

# Step 7: Testing the model with an Image

def preprocess_img(image_path: str):

    img = Image.open(image_path)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # converting Image to tensor because models work on tensors, not images
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean = mean, std = std)
    ])

    tensor = transform(img)
    tensor = tensor.unsqueeze(0)
    return tensor

IMAGE_PATH = "../Dataset/Processed_Images/667626_18933d713e.jpg"
processed_img = preprocess_img(IMAGE_PATH)
# print(processed_img.shape)

# Step 8: Passing the preprocessed image tensor to the model for inference
model.eval()
with torch.no_grad():
    output = model(processed_img)
print(output.shape)





