#%%

import pandas as pd
import glob
import os
import numpy as np
from PIL import Image


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
# Convert dataset to pytorch dataset



#%%
# Set up GAN

