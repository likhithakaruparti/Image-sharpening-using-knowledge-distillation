import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import BlurSharpDataset
from student_model import StudentCNN
import sys

# Add Restormer path for import
sys.path.append('Restormer')
from basicsr.models.archs.restormer_arch import Restormer

# ======== Setup ========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Paths
blur_path = 'data/blur'
sharp_path = 'data/sharp'
checkpoint_path = 'Restormer/weights/deraining.pth'

# ======== Load Dataset ========
dataset = BlurSharpDataset(blur_path, sharp_path)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
print(f"✅ Loaded dataset with {len(dataset)} image pairs")

# ======== Load Teacher Model (Restormer) ========
teacher = Restormer().to(device)
checkpoint = torch.load(checkpoint_path, map_location=device)
teacher.load_state_dict(checkpoint['params'])
teacher.eval()
print("✅ Loaded teacher model")

# ======== Initialize Student Model (CNN) ========
student = StudentCNN().to(device)
print("✅ Initialized student model")

# ======== Loss & Optimizer ========
criterion = nn.L1Loss()
optimizer = optim.Adam(student.parameters(), lr=1e-4)
print("✅ Loss function and optimizer set")

# ======== Full Training Loop ========
epochs = 2  # Reduced for fast training
print(f"\n🚀 Starting training for {epochs} epochs...\n")

for epoch in range(epochs):
    student.train()
    running_loss = 0.0

    print(f"\n📘 Epoch {epoch+1} started")

    for i, batch in enumerate(dataloader):
        print(f"🔁 Processing batch {i+1}/{len(dataloader)}")

        blurry = batch['blur'].to(device)

        with torch.no_grad():
            teacher_output = teacher(blurry)
        print("✅ Teacher output ready")

        student_output = student(blurry)
        print("✅ Student output ready")

        loss = criterion(student_output, teacher_output)
        print(f"📉 Loss: {loss.item():.4f}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(dataloader)
    print(f"✅ Epoch [{epoch+1}/{epochs}] - Average Loss: {avg_loss:.4f}")

# ======== Save the Trained Student Model ========
torch.save(student.state_dict(), 'student_model.pth')
print("\n🎉 Training complete! Student model saved as student_model.pth")