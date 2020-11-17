#%%

import pandas as pd
import glob

# path = r'C:\Users\gobli\Documents\AI CORE\Machine-Learning-Algos\ShoeGAN\server\data'
# all_files = glob.glob(os.path.join(path, "*.txt")) 

# df_from_each_file = [pd.read_csv(f) for f in all_files]

# concatenated_df = pd.concat(df_from_each_file, axis=0, ignore_index=True)
adidas_raw = pd.read_csv("data/adidas.txt", sep="\r")
adidas_raw.head()

#%%
import requests

r = requests.get(adidas_raw.iloc[1][0])
print(r)