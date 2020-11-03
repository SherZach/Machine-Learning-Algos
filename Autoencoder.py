import torch
import numpy as np
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

# Get train dataset
train_data = datasets.MNIST(root='MNIST-data',                        
                            transform=transforms.ToTensor(),          
                            train=True,                               
                            download=True                             
                           )

# Get test dataset
test_data = datasets.MNIST(root='MNIST-data',
                           transform=transforms.ToTensor(),
                           train=False,
                          )

# train_data, valid_data = torch.utils.data.random_split(train_data, [50000, 10000])

batch_size = 128

train_loader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=batch_size)
# valid_loader = torch.utils.data.DataLoader(valid_data, shuffle=True, batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(test_data, shuffle=True, batch_size=batch_size)

class Encoder(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(784, 64),
            torch.nn.ReLU(), # F.relu ?
            torch.nn.Linear(64, 16),
            # torch.nn.ReLU()
        )
        
    def forward(self, x):
        x = x.view(-1, 784)
        x = self.layers(x)
        return x

class Decoder(torch.nn.Module):
    
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(16, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 784),
            torch.nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.layers(x)
        return x

class AutoEncoder(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


autoenc = AutoEncoder()

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adagrad(autoenc.parameters(), lr=0.005)

writer = SummaryWriter(log_dir='../runs') 

def train(model, epochs):
    model.train()                                  
    for epoch in range(epochs):
        for idx, minibatch in enumerate(train_loader):
            inputs, _ = minibatch
            prediction = model(inputs)             
            loss = criterion(prediction, inputs.view(-1, 784))   
            print('Epoch:', epoch, '\tBatch:', idx, '\tLoss:', loss)
            optimizer.zero_grad()                  
            loss.backward()                        
            optimizer.step()
            writer.add_scalar('Loss/Train', loss, epoch*len(train_loader) + idx)                       

    t = transforms.ToPILImage()
    img = prediction[0].view(28, 28)
    t(img).show()
    t(inputs[0]).show()
    

train(autoenc, 2)
