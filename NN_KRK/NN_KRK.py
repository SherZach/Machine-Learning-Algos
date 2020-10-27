
import pandas as pd 
import torch
from torch.utils.data import Dataset

# Reading in dataset from its .csv formart to a Pandas dataframe
raw_data = pd.read_csv('NN_KRK\krkopt.data', names=["WKf", "WKr", "WRf", "WRr", "BKf", "BRr", "DofW"])
# print(raw_data)

# Data is all categorical so convert it to category, then to a number code
for column_name, item in raw_data.iteritems():
    raw_data[column_name] = raw_data[column_name].astype("category")
    raw_data[column_name] = raw_data[column_name].cat.codes

# view data
# print(raw_data.head())
# print(raw_data["DofW"].describe())

# Convert dataset from a dataframe to a pytorch dataset
class KRKDataset(Dataset):
    def __init__(self, data):
        self.features = data
        self.labels = self.features.pop('DofW')
        
    def __getitem__(self, i):
        return torch.tensor(self.features.iloc[i]), torch.tensor(self.labels[i])
    
    def __len__(self):
        return len(self.features)

data = KRKDataset(raw_data)

# test it is now performing as a pytorch dataset
print(len(data))
print(data[0])

