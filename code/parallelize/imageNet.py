import os 
import pickle 
import time
import json 

import click
from PIL import Image

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class ImageNet(Dataset):
    def __init__(self, root, split, transform=None):
        if split not in ["train", "val"]:
            raise ValueError("split must be either 'train' or 'val'")
        
        self.root = root
        
        with open(os.path.join(root, "imagenet_{}.pkl".format(split)), "rb") as f:
            data = pickle.load(f)

        self.samples = list(data.keys())
        self.targets = list(data.values())
        self.transform = transform
        
                
    def __len__(self):
        return len(self.samples)    
    
    def __getitem__(self, idx):
        x = Image.open(os.path.join(self.root, self.samples[idx])).convert("RGB")
        if self.transform:
            x = self.transform(x)
        return x, self.targets[idx]
