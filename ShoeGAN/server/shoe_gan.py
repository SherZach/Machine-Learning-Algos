#%%

import pandas as pd
import glob
import os

path = r'C:\Users\gobli\Documents\AI CORE\Machine-Learning-Algos\ShoeGAN\server\data'
all_files = glob.glob(os.path.join(path, "*.txt")) 

df_from_each_file = [pd.read_csv(f, sep="\r",names=["Link"]) for f in all_files]
# print(df_from_each_file)

total_df = pd.concat(df_from_each_file, ignore_index=True)
total_df
print(total_df.iloc[46][0])
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
from PIL import Image

# Convert all donwloaded images into PIL images
