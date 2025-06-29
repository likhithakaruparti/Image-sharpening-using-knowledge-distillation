import torch
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt
from student_model import StudentCNN
import sys

# Add Restormer path
sys.path.append('Restormer')
from basicsr.models.archs.restormer_arch import Restormer

# ======== Setup ========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======== Load Models ========
# Load trained student model
student = StudentCNN().to(device)
student.load_state_dict(torch.load("student_model.pth", map_location=device))
student.eval()

# Load Restormer (teacher) model
teacher = Restormer().to(device)
checkpoint = torch.load("Restormer/weights/deraining.pth", map_location=device)
teacher.load_state_dict(checkpoint["params"])
teacher.eval()

# ======== Load a Test Blurry Image ========
img_path = "data/blur/000001.png"  # or any image you want to test
img = Image.open(img_path).convert("RGB")

transform = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor()
])
input_tensor = transform(img).unsqueeze(0).to(device)

# ======== Generate Outputs ========
with torch.no_grad():
    teacher_output = teacher(input_tensor)
    student_output = student(input_tensor)

# ======== Convert to Images ========
def tensor_to_img(tensor):
    img = tensor.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy()
    return img

input_img = tensor_to_img(input_tensor)
teacher_img = tensor_to_img(teacher_output)
student_img = tensor_to_img(student_output)

# ======== Show Side-by-Side ========
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(input_img)
plt.title("🔸 Blurry Input")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(teacher_img)
plt.title("🔹 Teacher Output")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(student_img)
plt.title("🟢 Student Output")
plt.axis("off")

plt.tight_layout()
plt.show()
