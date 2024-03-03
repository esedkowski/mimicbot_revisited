import os
import torch
import pandas as pd
#from skimage import io, transform
#import numpy as np
#import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
#from torchvision import transforms, utils

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

class MimicDataset(Dataset):
    """Mimic dataset."""

    def __init__(self, csv_file, root_dir):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the arrays.
        """
        self.mimic = pd.read_csv(csv_file)
        self.root_dir = root_dir

    def __len__(self):
        return len(self.mimic)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        inputs_file = os.path.join(self.root_dir, self.mimic.iloc[idx, 0])
        inputs = torch.load(inputs_file)
        
        outputs_file = os.path.join(self.root_dir, self.mimic.iloc[idx, 1])
        outputs = torch.load(outputs_file)

        return (inputs, outputs)
    
mimic_dataset = MimicDataset(csv_file='.\\test\\data.csv',
                                    root_dir='')

mimic_dataloader = DataLoader(mimic_dataset, batch_size=1, shuffle=True, num_workers=0)


if __name__ == '__main__':
    
    for i, (inp, out) in enumerate(mimic_dataloader):
        inp['spatial_obs'].float
        print(type(inp['spatial_obs'][0]), inp['spatial_obs'][0].size())
        if i == 3:
            break
