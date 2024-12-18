import torch
import torchvision
import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pathlib import Path
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2

IMAGENET_DEFAULT_MEAN = [0.485, 0.456, 0.406]
IMAGENET_DEFAULT_STD = [0.229, 0.224, 0.225]
img_size = 224

images_path = Path('PH2/images') # resolver esta merda

data = pd.read_csv('PH2_dataset.csv')

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
main_dir = os.path.dirname('PH2_dataset.csv')


train_transforms = A.Compose(
            [
                A.Resize(width=img_size, height=img_size),
                A.HorizontalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ColorJitter(
                    brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8
                ),
                A.GaussianBlur(p=0.5),
                A.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
                ToTensorV2(),
            ]
        )
eval_transforms = A.Compose(
            [
                A.Resize(width=img_size, height=img_size),
                A.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
                ToTensorV2(),
            ]
        )



for fold, (test_set, train_set) in enumerate(skf.split(data, data['labels'])):
    # Get train and validation splits
    train_data = data.iloc[train_set]
    val_data = data.iloc[test_set]

    # Save to CSV
    train_path = os.path.join(main_dir, f'train_fold_{fold + 1}.csv')
    val_path = os.path.join(main_dir, f'val_fold_{fold + 1}.csv')
    train_data.to_csv(train_path, index=False)
    val_data.to_csv(val_path, index=False) 


class PH2Dataset(Dataset):
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
    



for fold in range(5):
    train_csv = os.path.join(main_dir, f'train_fold_{fold + 1}.csv')
    test_csv = os.path.join(main_dir, f'val_fold_{fold + 1}.csv')

    train_dataset = PH2Dataset(train_csv, img_dir=images_path, transform=train_transforms)
    test_dataset = PH2Dataset(test_csv, img_dir=images_path, transform=eval_transforms)

    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=16,
                                  shuffle=True,
                                  num_workers=os.cpu_count())
    
    test_dataloader = DataLoader(dataset=test_dataset,
                                  batch_size=16,
                                  shuffle=False,
                                  num_workers=os.cpu_count())
     




