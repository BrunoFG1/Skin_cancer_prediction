import torch
import torchvision
import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset, DataLoader
from PIL import Image

data = pd.read_csv('PH2_dataset.csv')

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
main_dir = os.path.dirname('PH2_dataset.csv')


for fold, (test_set, train_set) in enumerate(skf.split(data, data['labels'])):
    # Get train and validation splits
    train_data = data.iloc[train_set]
    val_data = data.iloc[test_set]

    # Save to CSV
    train_path = os.path.join(main_dir, f'train_fold_{fold + 1}.csv')
    val_path = os.path.join(main_dir, f'val_fold_{fold + 1}.csv')
    train_data.to_csv(train_path, index=False)
    val_data.to_csv(val_path, index=False) 


class CustomDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
       
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        
        img_name = self.data.iloc[index, 0]  
        label = self.data.iloc[index, 1]    
        
        
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label 
    
     


