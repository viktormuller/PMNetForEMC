import pandas as pd
import torch
from torchvision import transforms
from torch.utils.data import Dataset
import os
import numpy as np
from skimage import io
from torch.utils.data import DataLoader
import torch
from torch.utils.data import Dataset, DataLoader, random_split


class PMNet_USC_per_pixel(Dataset): 
    def __init__(self, csv_file,
                 dir_dataset="data/USC/",               
                 transform= transforms.ToTensor()):
        self.ind_val = pd.read_csv(csv_file)
        self.dir_dataset = dir_dataset
        self.transform = transform

    def __len__(self):
        return len(self.ind_val)
    
    def __getitem__(self, idx):

        #Load city map
        self.dir_buildings = self.dir_dataset+ "/map/"
        img_name_buildings = os.path.join(self.dir_buildings, str((self.ind_val.iloc[idx, 0]))) + ".png"
        image_buildings = np.asarray(io.imread(img_name_buildings))
        
        positions = torch.tensor(self.ind_val.iloc[idx,1:3].values, dtype=torch.float32)
        powers = torch.tensor(self.ind_val.iloc[idx,3], dtype=torch.float32)
      
        inputs = (positions, image_buildings)

        if self.transform:            
            image_buidlings = self.transform(image_buildings).type(torch.float32)            

        inputs = (positions, image_buildings)
            
        return [inputs , powers]
    

def getPMNet_USC_PP_DataLoaders(csv_file, batch_size=128, test_batch_size=512):
    per_pixel_dataset = PMNet_USC_per_pixel(csv_file=csv_file)

    # Set a fixed seed for reproducibility
    seed = 42
    torch.manual_seed(seed)

    # Define the sizes for train, test, and validation sets
    total_samples = len(per_pixel_dataset)
    train_size = int(0.8 * total_samples)
    test_size = int(0.1 * total_samples)
    val_size = total_samples - train_size - test_size

    # Use random_split to split the dataset
    train_dataset, test_dataset, val_dataset = random_split(
        per_pixel_dataset, [train_size, test_size, val_size], generator=torch.Generator().manual_seed(seed)
    )

    # Print sizes of the split datasets
    print("Training set size:", len(train_dataset))
    print("Testing set size:", len(test_dataset))
    print("Validation set size:", len(val_dataset))


    train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dl = DataLoader(test_dataset, batch_size=test_batch_size)
    validate_dl = DataLoader(val_dataset, batch_size=test_batch_size)

    return {
        'train': train_dl,
        'test': test_dl,
        'validate': validate_dl
    }