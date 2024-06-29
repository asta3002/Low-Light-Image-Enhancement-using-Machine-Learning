import numpy as np
import cv2
from Components import *
import os
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from PIL import Image

class SICEDataset(Dataset):
    def __init__(self, data_folder, transform=None):
        self.data_folder = data_folder
        self.image_files = [f for f in os.listdir(data_folder) if f.endswith(('.png', '.jpg', '.jpeg', '.gif'))]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        image_path = os.path.join(self.data_folder, image_file)

        # Load image
        image = Image.open(image_path).convert('RGB')  # Convert to RGB if needed

        # Apply transformations if available
        if self.transform:
            image = self.transform(image)

        return image

def Get_Data(data_folder):

    transform = transforms.Compose([ transforms.ToTensor(),])

    # Create the dataset instance
    dataset = SICEDataset(data_folder, transform=transform)

    batch_size = 32

    # Create the data loader
    train_data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return train_data_loader

def Extract_Channels(batch,device):
    R = batch[:,0,:,:].unsqueeze(1)
    G = batch[:,1,:,:].unsqueeze(1)
    B  =batch[:,2,:,:].unsqueeze(1)
    RG = torch.cat((R,G),dim=1).to(device)
    RB  = torch.cat((R,B),dim=1).to(device)
    GB = torch.cat((G,B),dim=1).to(device)
    return RG,RB,GB

    