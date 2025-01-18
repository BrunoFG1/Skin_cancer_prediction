import torch
import torchvision
from torch import nn
import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pathlib import Path
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt 
import numpy as np
import timm
from going_modular.going_modular import engine, utils
from helper_functions import plot_loss_curves
import json
from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix
from tqdm.auto import tqdm

IMAGENET_DEFAULT_MEAN = [0.485, 0.456, 0.406]
IMAGENET_DEFAULT_STD = [0.229, 0.224, 0.225]
img_size = 224

torch.manual_seed(42)

device = "cuda" if torch.cuda.is_available() else "cpu"

images_path = Path('PH2/images') # resolver esta merda

data = pd.read_csv('data/PH2_dataset.csv')

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
main_dir = os.path.dirname('data/PH2_dataset.csv')


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
        
        img_path = os.path.join(self.img_dir, img_name + ".jpg")
        image = Image.open(img_path).convert("RGB")


        if self.transform:
            aug = self.transform(image=np.array(image))
            image = aug['image']
        
        return image, label 
    

fold_Accuracy = []
results = {}

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
    
    '''for batch in train_dataloader:
        images, labels = batch

        images = images.permute(0, 2, 3, 1).cpu().numpy()

        fig, axs = plt.subplots(4, 4, figsize=(12, 12)) 
        axs = axs.flatten()

        for img, label, ax in zip(images, labels, axs):
            ax.imshow(img)
            ax.set_title(f"Label: {label}")
            ax.axis('off')

        
        plt.show()'''

    # Create model
    model1 = timm.create_model('densenet121', pretrained = True, num_classes=2)
    model2 = timm.create_model('resnet101', pretrained=True, num_classes =2)
    model3 = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes =2)
    model = timm.create_model('resnet50', pretrained=True, num_classes = 2)

    loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                        mode='min',
                                                        factor=0.1)
    
    '''model_results = engine.train(model=model,
                             train_dataloader=train_dataloader,
                             test_dataloader=test_dataloader,
                             optimizer=optimizer,
                             loss_fn=loss_fn,
                             epochs=100,
                             device=device,
                             scheduler=scheduler)
    
    accuracy = model_results["test_acc"][-1]
    print(f"{fold} acc is:  {accuracy}  ")
    fold_Accuracy.append(accuracy)

utils.save_model(model=model,
                 target_dir='models/',
                 model_name='vit_base_patch16_224.pth')'''

models_info = {
    "densenet121_model": model1,
    "resnet101": model2,
    "vit_base_patch16_224": model3,
    "best_model": model
}

sum_num = 0
for i in fold_Accuracy:
    sum_num += i

# Carregar os resultados do arquivo
with open("assets/model_results.json", "r") as f:
    results = json.load(f)

'''average_acc = sum_num/len(fold_Accuracy)
results['resnet50'] = average_acc

with open("model_results.json", "w") as f:
    json.dump(results, f, indent=4)'''


#print(f"Average accuracy: {average_acc}")
# Convert the results dictionary to a DataFrame
df = pd.DataFrame.from_dict(results, orient="index", columns=["Accuracy"])

# Add a column for model names
df.index.name = "Model"
df.reset_index(inplace=True)

sens_list = []
spec_list = []
ball_acc = []

for model_name, model in models_info.items():
    model_path = f'models/{model_name}.pth'  # Use the key (string) to create the path
    loaded_model = model  # Assign the actual model object
    loaded_model.load_state_dict(torch.load(model_path))  # Load the weights
    loaded_model.to(device)
    loaded_model.eval()

    y_preds = []
    y_true = []

    with torch.inference_mode():
        for X, y in tqdm(test_dataloader, desc="Making predictions"):
        
            X, y = X.to(device), y.to(device)

            y_logit = loaded_model(X)
            # Turn predictions from logits -> prediction probabilities -> predictions labels
            y_pred = torch.softmax(y_logit, dim=1).argmax(dim=1) 

            y_true.append(y.cpu())
            y_preds.append(y_pred.cpu())

    # Concatenate list of predictions into a tensor
    y_pred_tensor = torch.cat(y_preds)
    y_true_tensor = torch.cat(y_true)

    # Confusion Matrix
    confmat = ConfusionMatrix(task='binary', num_classes=2)
    conf_mtx = confmat(y_pred_tensor, y_true_tensor)

    # Plot the confusion matrix
    fig, ax = plot_confusion_matrix(
        conf_mat=conf_mtx.numpy(),
        class_names=['0', '1'],
        figsize=(10, 7)
    )

    TN = conf_mtx[0,0]
    FN = conf_mtx[0,1]
    FP = conf_mtx[1,0]
    TP = conf_mtx[1,1]

    sensivity = TP/(TP + FN)
    specificity = TN/(TN + FP)
    ballance_acc = (sensivity + specificity)/2
    ball_acc.append(ballance_acc)

    print(f"Sensivity: {sensivity}")
    print(f"specificity: {specificity}")

    sens_list.append(sensivity)
    spec_list.append(specificity)

# Add column in tabel to compare Sensivity and specificity
df["Sensivity"] = sens_list
df["Specificity"] = spec_list
df["Balanced Accuracy"] = ball_acc


# Print the table
print(df)
