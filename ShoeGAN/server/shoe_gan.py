#%%

import pandas as pd
import glob
import os
import numpy as np
from PIL import Image
import torch
from torchvision import transforms as T
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt  
from torch.utils.tensorboard import SummaryWriter
from time import time

#%%

path = r'C:\Users\gobli\Documents\AI CORE\Machine-Learning-Algos\ShoeGAN\server\data'
all_files = glob.glob(os.path.join(path, "*.txt")) 

df_from_each_file = [pd.read_csv(f, sep="\r",names=["Link"]) for f in all_files]
# print(df_from_each_file)

total_df = pd.concat(df_from_each_file, ignore_index=True)
total_df
# print(total_df.iloc[46][0])
#%%
import requests

for i in range(len(total_df)):
    try:
        r = requests.get(total_df.iloc[i][0], timeout=10)
    except:
        print("Problem:", i)
        continue

    if r.ok:
        print("Sucess:", i)
        with open("data/images/" + str(i) + ".jpg", "wb") as f:
            f.write(r.content)
# Do not run this cell again unless you want to waste 10 minutes!
#%%
# preprocess images

img_list = []
directory = r'C:\Users\gobli\Documents\AI CORE\Machine-Learning-Algos\ShoeGAN\server\data\images'
for entry in os.scandir(directory):
    if entry.path.endswith(".jpg") and entry.is_file():
        try: 
            img = Image.open(entry.path) 
        except: 
            continue
        # convert all donwloaded images into PIL images
        img = img.convert()
        img_list.append(img)
        
# get the list of sizes of the images
size_list = [img.size for img in img_list]
# get the median of the sizes
med_width, med_height = np.median(np.array(size_list), axis=0)
print(med_width, med_height)
# resize all images to be equal to the average size
img_list = [img.resize((584, 584)) for img in img_list]
#%%
# Convert to a list of tensors
transform = T.ToTensor()
img_tensors = [transform(img) for img in img_list]

#%%
# get data loader
batch_size = 16
latent_vec_size = 4000

train_loader = DataLoader(img_tensors, shuffle=True, batch_size=batch_size)
for i in range(len(img_tensors)):
    if img_tensors[i].shape==torch.Size([4, 584, 584]):
        print(True)
        img_tensors[i].pop()  # LOOK AT THIS!!!!!!!!

#%%
# Set up discriminator
class Discriminator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=4, stride=6, padding=1) #1x28x28-> 64x14x14
        self.conv2 = torch.nn.Conv2d(64, 128, kernel_size=9, stride=7, padding=1) #64x14x14-> 128x7x7
        self.dense1 = torch.nn.Linear(128*14*14, 1)

        self.bn1 = torch.nn.BatchNorm2d(64)
        self.bn2 = torch.nn.BatchNorm2d(128)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x))).view(-1, 128*14*14)
        x = F.sigmoid(self.dense1(x))
        return x

# Set up generator
class Generator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.x = 14
        self.dense1 = torch.nn.Linear(latent_vec_size, 512)
        self.dense2 = torch.nn.Linear(512, 1024)
        self.dense3 = torch.nn.Linear(1024, 128*self.x**2)
        self.uconv1 = torch.nn.ConvTranspose2d(128, 64, kernel_size=9, stride=7, padding=1) #128x7x7 -> 64x14x14
        self.uconv2 = torch.nn.ConvTranspose2d(64, 3, kernel_size=4, stride=6, padding=1) #64x14x14 -> 1x28x28

        self.bn1 = torch.nn.BatchNorm1d(512)
        self.bn2 = torch.nn.BatchNorm1d(1024)
        self.bn3 = torch.nn.BatchNorm1d(128*self.x**2)
        self.bn4 = torch.nn.BatchNorm2d(64)
    
    def forward(self, x, latent_vec_size):
        x = F.relu(self.bn1(self.dense1(x)))
        x = F.relu(self.bn2(self.dense2(x)))
        x = F.relu(self.bn3(self.dense3(x))).view(-1, 128, self.x, self.x)
        x = F.relu(self.bn4(self.uconv1(x)))
        x = F.sigmoid(self.uconv2(x))
        return x

# g = Generator()
# d = Discriminator()


# ran_batch = torch.rand(batch_size, latent_vec_size)
# # ran_batch.to(device)
# fake = g(ran_batch, latent_vec_size)
# print(fake.shape)
# output = d(fake)
# print(output.shape)

#%%

# function to show image from tensor
def img_show(tens):
    tens = torch.squeeze(tens)
    print(tens.shape)
    trans = T.ToPILImage()
    img = trans(tens)
    img.show()

img_show(fake[4])
# ADJUST VALUES TO GE THE RIGHT SHAPE: (3, 580, 600)
#%%
#instantiate model
d = Discriminator()
g = Generator()

#training hyperparameters
no_epochs = 100
dlr = 0.0003
glr = 0.0003

d_optimizer = torch.optim.Adam(d.parameters(), lr=dlr)
g_optimizer = torch.optim.Adam(g.parameters(), lr=glr)

dcosts = []
gcosts = []
plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlabel('Epoch')
ax.set_ylabel('Cost')
ax.set_xlim(0, no_epochs)
plt.show()
# %%

# Training loop 

device = "cuda:0" if torch.cuda.is_available() else "cpu"
device = "cpu"

def train(G, D, epochs=1):
    optimiser_d = torch.optim.Adam(D.parameters(), lr=0.00001)
    optimiser_g = torch.optim.Adam(G.parameters(), lr=0.0001)
    writer = SummaryWriter(log_dir=f'runs/DCGAN/{time()}')
    G = G.to(device)
    D = D.to(device)
    batch_idx = 0
    
    for epoch in range(epochs):
        for idx, (x, _) in enumerate(train_loader):
            x = x.to(device)
            print(x)
            z = torch.randn(batch_size, latent_vec_size)
            z = z.to(device)

            # GENERATOR UPDATE
            optimiser_g.zero_grad()
            labels = torch.ones(batch_size).to(device)
            G_loss = criterion(D(G(z)), labels)
            # G_loss = - torch.log(1 - D(G(z)))
            # G_loss = torch.mean(G_loss)
            G_loss.backward(retain_graph=True)
            optimiser_g.step()

            # DISCRIMINATOR UPDATE
            optimiser_d.zero_grad()
            labels = torch.zeros(batch_size).to(device)
            D_loss = criterion(D(G(z)), labels) # loss on fake examples
            # D_loss = - (torch.log(D(x)) + torch.log(1 - D(G(z))))
            # D_loss = torch.mean(D_loss)
            D_loss.backward()
            labels = torch.ones(x.shape[0]).to(device)
            D_loss = criterion(D(x), labels)
            D_loss.backward()
            optimiser_d.step()
            
            writer.add_scalar('Loss/G', G_loss.item(), batch_idx)
            writer.add_scalar('Loss/D', D_loss.item(), batch_idx)
            batch_idx += 1
            print(
                'Epoch:', epoch ,
                'Batch:', idx,
                'Loss G:', G_loss.item(),
                'Loss D:', D_loss.item()
            )
            if idx % 100 == 0:
                print('sampling')
                sample(writer, device)

G = Generator()
D = Discriminator()
train(G, D, epochs=10)