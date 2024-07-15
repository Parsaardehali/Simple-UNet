import torch
from torch.utils.data import Dataset

# Dummy dataset
class DummyDataset(Dataset):
    def __init__(self, num_samples=100, img_size=64):
        self.num_samples = num_samples
        self.img_size = img_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate dummy input and target
        input_img = torch.rand(1, self.img_size, self.img_size)
        target_img = torch.rand(1, self.img_size, self.img_size)
        return input_img, target_img

# Proper Dataset
class SegmentationDataset(Dataset):
    def __init__(self, num_samples=100, img_size=64):
        self.num_samples = num_samples
        self.img_size = img_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate a background
        input_img = torch.rand(1, self.img_size, self.img_size)

        # Create a simple shape for segmentation (e.g., a circle)
        x = torch.arange(self.img_size).unsqueeze(1).repeat(1, self.img_size).float()
        y = torch.arange(self.img_size).unsqueeze(0).repeat(self.img_size, 1).float()
        center = self.img_size // 2
        radius = torch.randint(self.img_size // 8, self.img_size // 4, (1,)).item()
        mask = ((x - center)**2 + (y - center)**2 <= radius**2).float()

        # Add the shape to the input image
        input_img[0] += mask * 0.5
        input_img = torch.clamp(input_img, 0, 1)

        # Create the target segmentation mask
        target_img = mask.unsqueeze(0)

        return input_img, target_img
