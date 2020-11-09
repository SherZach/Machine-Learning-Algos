#%%
import pandas as pd
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F 

# read in data
raw_data = pd.read_csv("BC_gene_data.csv")
# print(raw_data.head())

#%%

# convert class labels to categories and to int64s
raw_data["type"] = raw_data["type"].astype("category").cat.codes
raw_data["type"] = raw_data["type"].astype("int64")
print(raw_data["type"].unique())
print(raw_data.shape)

#%%
# convert to a pytorch dataset
class BCGeneDataset(Dataset):
    def __init__(self,data):
        self.features = data
        self.labels = self.features.pop("type")

    def __getitem__(self, idx):
        features, labels = torch.tensor(self.features.iloc[idx]), torch.tensor(self.labels[idx])
        return features, labels
    
    def __len__(self):
        return len(self.features)

data = BCGeneDataset(raw_data)
print(data[10])
print(len(data))
# print(data)

#%%
# split into test, train, validation
train_data, valid_data, test_data = torch.utils.data.random_split(data, [121, 15, 15])

# Create dataloaders
batch_size = 30

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

#%%
# create NN class

class NN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Linear(54676, 54676)
        self.layer2 = torch.nn.Linear(54676, 54676/4)
        self.layer3 = torch.nn.Linear(54676/4, 54676/ 128)
        self.layer4 = torch.nn.Linear(54676/128, 6)

    def forward(self, x):
        x = x.to('cuda')
        x = x.flatten()
        x = self.layer1(x)
        x = F.relu(x)
        x = self.layer2(x)
        x = F.relu(x)
        x = self.layer3(x)
        x = F.relu(x)
        x = self.layer4(x)
        x = F.softmax(x)
        return x

nn = NN()

#%%
# Train the NN
learning_rate = 0.1

# Create the optimiser (Adam)
optimiser = torch.optim.Adam(              
    nn.parameters(),          
    lr=learning_rate                      
)

# Create our criterion
criterion = torch.nn.CrossEntropyLoss()  