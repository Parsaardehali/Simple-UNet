import torch
from torch.utils.data import Dataset

import numpy as np

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
        x, y = np.ogrid[:self.img_size, :self.img_size]
        center = self.img_size // 2
        radius = np.random.randint(self.img_size // 8, self.img_size // 4)
        mask = ((x - center)**2 + (y - center)**2 <= radius**2).astype(float)
        
        # Add the shape to the input image
        input_img[0] += torch.from_numpy(mask) * 0.5
        input_img = torch.clamp(input_img, 0, 1)
        
        # Create the target segmentation mask
        target_img = torch.from_numpy(mask).float().unsqueeze(0)
        
        return input_img, target_img

# Training function
def train(model, dataloader, criterion, optimizer, device, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(dataloader):.4f}")