import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class BlurSharpDataset(Dataset):
    def __init__(self, blur_dir, sharp_dir, image_size=(256, 256)):
        self.blur_dir = blur_dir
        self.sharp_dir = sharp_dir
        self.image_filenames = sorted(os.listdir(blur_dir))
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        blur_path = os.path.join(self.blur_dir, self.image_filenames[idx])
        sharp_path = os.path.join(self.sharp_dir, self.image_filenames[idx])

        blur_image = Image.open(blur_path).convert("RGB")
        sharp_image = Image.open(sharp_path).convert("RGB")

        return {
            "blur": self.transform(blur_image),
            "sharp": self.transform(sharp_image)
        }