import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import datasets
import pdb

class ProvoDataset(Dataset):
    def __init__(self, seqs):
        self.seqs = seqs.reset_index(drop=True)
    
    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return self.seqs.iloc[idx]

