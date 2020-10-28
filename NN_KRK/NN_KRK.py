
import pandas as pd 
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F 

# Reading in dataset from its .csv formart to a Pandas dataframe
raw_data = pd.read_csv('NN_KRK\krkopt.data', names=["WKf", "WKr", "WRf", "WRr", "BKf", "BRr", "DofW"])
# print(raw_data)

# Data is all categorical so convert it to category, then to a number code
for column_name, item in raw_data.iteritems():
    raw_data[column_name] = raw_data[column_name].astype("category")
    raw_data[column_name] = raw_data[column_name].cat.codes

# view data
print("A sample of the dataset\n", raw_data.head())
# print(raw_data["DofW"].describe())
input("Press ENTER to continue...")

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
print("Testing the dataset is now pytorch compatible")
print(len(data))
print(data[0])
input("Press ENTER to continue...")

# Split into train, test and validation

train_data, test_data = torch.utils.data.random_split(data, [22056, 6000])
train_data, valid_data = torch.utils.data.random_split(train_data, [16056, 6000])

# Create dataloaders
batch_size = 200

train_loader = torch.utils.data.DataLoader( 
    train_data, 
    shuffle=True, 
    batch_size=batch_size
)

valid_loader = torch.utils.data.DataLoader(
    valid_data,
    shuffle=True,
    batch_size=batch_size
)

test_loader = torch.utils.data.DataLoader(
    test_data, 
    shuffle=True, 
    batch_size=batch_size
)

# Create NN class

class NN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(6, 600),
            F.relu(),
            torch.nn.Linear(600, 100),
            F.relu(),
            torch.nn.Linear(100,18),
        )
        self.final_layer = torch.nn.Softmax()

    def forward(self, x):
        x = self.layers(x)
        x = self.final_layer(x)
        return x

nn = NN()