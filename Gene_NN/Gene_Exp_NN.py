#%%
import pandas as pd
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F 
import sklearn.metrics

# read in data
raw_data = pd.read_csv("BC_gene_data.csv")
# print(raw_data.head())

#%%

raw_data.drop(labels="samples", axis=1, inplace=True)
# convert class labels to categories and to int64s, convert rest of data to float32
raw_data["type"] = raw_data["type"].astype("category").cat.codes
raw_data = raw_data.astype("float32")
raw_data["type"] = raw_data["type"].astype("int64")
print(raw_data["type"].unique())
print(raw_data.shape)

# print(raw_data["1007_s_at"].dtype)
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
batch_size = 23

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
        self.layer1 = torch.nn.Linear(len(raw_data.columns), 2048)
        self.batch1 = torch.nn.BatchNorm1d(2048)
        self.layer2 = torch.nn.Linear(2048, 512)
        self.batch2 = torch.nn.BatchNorm1d(512)
        self.layer3 = torch.nn.Linear(512, 128)
        self.batch3 = torch.nn.BatchNorm1d(128)
        self.layer4 = torch.nn.Linear(128, 6)

    def forward(self, x):
        x = x.view(-1, len(raw_data.columns))
        x = F.relu(self.batch1(self.layer1(x)))
        x = F.relu(self.batch2(self.layer2(x)))
        x = F.relu(self.batch3(self.layer3(x)))
        x = F.softmax(self.layer4(x))
        return x



#%%
nn = NN() # Initialise NN
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #train on gpu
# nn.to(device)
learning_rate = 0.01 # Set learning rate

# Create the optimiser (Adam)
optimiser = torch.optim.Adam(              
    nn.parameters(),          
    lr=learning_rate                      
)

# Create our criterion
criterion = torch.nn.CrossEntropyLoss()  

# Train the model
def train(model, epochs):
    model.train()                                  # training mode
    for epoch in range(epochs):
        for idx, minibatch in enumerate(train_loader):
            inputs, labels = minibatch
            prediction = model(inputs)             # pass the data forward through the model
            loss = criterion(prediction, labels)   # compute the loss
            print('Epoch:', epoch, '\tBatch:', idx, '\tLoss:', loss)
            optimiser.zero_grad()                  # reset the gradients attribute of each of the model's params to zero
            loss.backward()                        # backward pass to compute and set all of the model param's gradients
            optimiser.step()                       # update the model's parameters

train(nn, 8)

#%%
# test on validation data
            
def test(model, dataloader, dataset):
    num_correct = 0
    num_examples = len(dataset)                       # test DATA not test LOADER
    for inputs, labels in dataloader:                  # for all exampls, over all mini-batches in the test dataset
        predictions = model(inputs)
        predictions = torch.max(predictions, axis=1)    # reduce to find max indices along direction which column varies
        predictions = predictions[1]                    # torch.max returns (values, indices)
        num_correct += int(sum(predictions == labels))
    percent_correct = num_correct / num_examples * 100
    print('Accuracy:', percent_correct)
    
test(nn, valid_loader, valid_data)
test(nn, test_loader, test_data)
# NEED F1 SCORE!!
