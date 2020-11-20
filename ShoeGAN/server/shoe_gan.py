#%%

import pandas as pd
import glob
import os
import numpy as np
from PIL import Image
import torch
from torchvision import transforms as T
from torch.utils.data import DataLoader


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
# resize all images to be equal to the average size
img_list = [img.resize((int(med_width), int(med_height))) for img in img_list]
#%%
# Convert to a list of tensors
transform = T.ToTensor()
img_tensors = [transform(img) for img in img_list]

#%%
# get data loader

train_loader = DataLoader(img_tensors, shuffle=True, batch_size=16)


#%%
# Set up discriminator
class discriminator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1) #1x28x28-> 64x14x14
        self.conv2 = torch.nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1) #64x14x14-> 128x7x7
        self.dense1 = torch.nn.Linear(128*7*7, 1)

        self.bn1 = torch.nn.BatchNorm2d(64)
        self.bn2 = torch.nn.BatchNorm2d(128)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x))).view(-1, 128*7*7)
        x = F.sigmoid(self.dense1(x))
        return x

# Set up generator
class generator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dense1 = torch.nn.Linear(128, 256)
        self.dense2 = torch.nn.Linear(256, 1024)
        self.dense3 = torch.nn.Linear(1024, 128*7*7)
        self.uconv1 = torch.nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1) #128x7x7 -> 64x14x14
        self.uconv2 = torch.nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1) #64x14x14 -> 1x28x28

        self.bn1 = torch.nn.BatchNorm1d(256)
        self.bn2 = torch.nn.BatchNorm1d(1024)
        self.bn3 = torch.nn.BatchNorm1d(128*7*7)
        self.bn4 = torch.nn.BatchNorm2d(64)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.dense1(x)))
        x = F.relu(self.bn2(self.dense2(x)))
        x = F.relu(self.bn3(self.dense3(x))).view(-1, 128, 7, 7)
        x = F.relu(self.bn4(self.uconv1(x)))
        x = F.sigmoid(self.uconv2(x))
        return x

