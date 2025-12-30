import torch
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import os

class FakeNewsDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.data.iloc[idx]['image_path']
        caption = str(self.data.iloc[idx]['caption'])
        label = int(self.data.iloc[idx]['label'])

        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)
        

        
        sample = {
            'image': image, 
            'text': caption, 
            'label': label,
            'image_path': img_path
        }

        return sample
