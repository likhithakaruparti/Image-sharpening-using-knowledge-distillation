import os
import torch
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt
import sys

# Add Restormer folder to system path
sys.path.append('Restormer')

# Import the model definition
from Restormer.basicsr.models.archs.restormer_arch import Restormer

# Load pretrained Restormer model
model = Restormer()
checkpoint_path = 'Restormer/weights/deraining.pth'
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint['params'])
model.eval()

# Load a sample blurry image
img_path = 'data/blur/000001.png'  # You can change this to any blurry image you have
img = Image.open(img_path).convert('RGB')

# Preprocess image
transform = T.Compose([
    T.Resize((256, 256)),     # Resize to what your student will use
    T.ToTensor()
])
input_tensor = transform(img).unsqueeze(0)  # Add batch dimension

# Pass through the teacher model
with torch.no_grad():
    output = model(input_tensor)

# Convert output tensor to image
output_img = output.squeeze(0).permute(1, 2, 0).clamp(0, 1).numpy()

# Show input and output side by side
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(img)
plt.title("Blurry Input")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(output_img)
plt.title("Teacher Output (Restormer)")
plt.axis('off')

plt.tight_layout()
plt.show()
