
import pandas as pd 
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F 
from torch.utils.tensorboard import SummaryWriter
import numpy as np

# Reading in dataset from its .csv formart to a Pandas dataframe
raw_data = pd.read_csv('NN_KRK\krkopt.data', names=["WKf", "WKr", "WRf", "WRr", "BKf", "BRr", "DofW"])
# print(raw_data)

# Data is all categorical so convert it to category, then to a number code
for column_name, item in raw_data.iteritems():
    raw_data[column_name] = raw_data[column_name].astype("category")
    raw_data[column_name] = raw_data[column_name].cat.codes
    raw_data[column_name] = raw_data[column_name].astype("float32")

raw_data["DofW"] = raw_data["DofW"].astype("int64")
# view data
print("A sample of the dataset\n", raw_data.head())
print(raw_data.describe())
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

train_data, valid_data, test_data = torch.utils.data.random_split(data, [16056, 6000, 6000])

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
        self.layer1 = torch.nn.Linear(6, 600)
        self.layer2 = torch.nn.Linear(600, 100)
        self.layer3 = torch.nn.Linear(100,18)

    def forward(self, x):
        x = x.view(-1, 6)
        x = self.layer1(x)
        x = F.relu(x)
        x = self.layer2(x)
        x = F.relu(x)
        x = self.layer3(x)
        x = F.softmax(x)
        return x

nn = NN()

# Train the NN
learning_rate = 1

# Create the optimiser (Adam)
optimiser = torch.optim.Adam(              
    nn.parameters(),          
    lr=learning_rate                      
)
# Create our criterion
criterion = torch.nn.CrossEntropyLoss()  
# Setup training visualisation
writer = SummaryWriter(log_dir='../runs')  

def train(model, epochs):
    model.to("cuda") # run on gpu
    model.train()                                  # put the model into training mode (more on this later)
    for epoch in range(epochs):
        for idx, minibatch in enumerate(train_loader):
            inputs, labels = minibatch
            prediction = model(inputs)             # pass the data forward through the model
            loss = criterion(prediction, labels)   # compute the loss
            print('Epoch:', epoch, '\tBatch:', idx, '\tLoss:', loss)
            optimiser.zero_grad()                  # reset the gradients attribute of each of the model's params to zero
            loss.backward()                        # backward pass to compute and set all of the model param's gradients
            optimiser.step()                       # update the model's parameters
            writer.add_scalar('Loss/Train', loss, epoch*len(train_loader) + idx)    # write loss to a graph

train(nn, 8)

# test on validation data
            
def test(model, dataloader):
    num_correct = 0
    num_examples = len(test_data)                       # test DATA not test LOADER
    for inputs, labels in dataloader:                  # for all exampls, over all mini-batches in the test dataset
        predictions = model(inputs)
        predictions = torch.max(predictions, axis=1)    # reduce to find max indices along direction which column varies
        predictions = predictions[1]                    # torch.max returns (values, indices)
        num_correct += int(sum(predictions == labels))
    percent_correct = num_correct / num_examples * 100
    print('Accuracy:', percent_correct)
    
test(nn, valid_loader)