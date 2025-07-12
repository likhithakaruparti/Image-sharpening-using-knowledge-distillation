# Image-sharpening-using-knowledge-distillation
Description: This project improves blurred images using a lightweight student CNN trained through knowledge distillation from a powerful transformer-based model, Restormer. The goal is to reduce model size while maintaining high image sharpening quality, making the solution suitable for deployment on real-time or low-power devices.
Problem Statement: High-performing models like Restormer provide excellent image sharpening but are too large for low-resource environments. This project uses knowledge distillation to train a compact CNN that mimics the teacher’s output and runs efficiently on lightweight systems.
Solution: A teacher-student setup is used where Restormer (teacher) produces sharp images, and a custom CNN (student) learns to replicate its outputs. No ground-truth labels are needed — the student learns only from the teacher. The student is then used alone for image sharpening.
Technologies used: Python - Core language  
PyTorch - Model training and architecture  
Torchvision - Transformations and DataLoader  
PIL, OpenCV - Image processing  
Matplotlib, Skimage - Visualization and evaluation  
VS Code - Development environment  
This project is purely backend-oriented.

Restormer dependancy: This project uses the Restormer model from GitHub:
Clone it inside your project folder:
git clone https://github.com/swz30/Restormer.git
Download the weights and place them at:
Restormer/weights/deraining.pth

Folder Structure: 
data/
├── blur/
└── sharp/
Restormer/
├── basicsr/
├── weights/
├── ...
scripts:
├── train_student.py
├── evaluate_metrics.py
├── visualize_results.py
├── dataset.py
├── student_model.py
├── teacher_test.py

Setup Instructions: 
# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install required packages
pip install -r requirements.txt

# Train the student model
python train_student.py

# Evaluate the trained model
python evaluate_metrics.py
