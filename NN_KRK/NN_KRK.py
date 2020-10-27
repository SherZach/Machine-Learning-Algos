import pandas as pd 
import torch
from torch.utils.data import Dataset

raw_data = pd.read_csv('NN_KRK\krkopt.data', names=["WKf", "WKr", "WRf", "WRr", "BKf", "BRr", "DofW"])
# print(raw_data)

class KRKDataset(Dataset):
    def __init__(self, data):
        self.features = data
        self.labels = self.features.pop("DofW")

data = KRKDataset(raw_data)

print(data.labels)
