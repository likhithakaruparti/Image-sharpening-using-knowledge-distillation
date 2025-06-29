from dataset import BlurSharpDataset
from torch.utils.data import DataLoader

# Your dataset paths
blur_path = 'data/blur'
sharp_path = 'data/sharp'

# Load the dataset
dataset = BlurSharpDataset(blur_path, sharp_path)

# Load images in batches
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Print shapes of one batch
for batch in dataloader:
    print("Blurry images shape:", batch['blur'].shape)
    print("Sharp images shape:", batch['sharp'].shape)
    break
