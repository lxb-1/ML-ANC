import os
import numpy as np
import pandas as pd
from tqdm import tqdm
pd.set_option('display.max_colwidth', -1)
pd.set_option('display.max_rows', None)

path = "/content/drive/MyDrive/Audio Signal Processing/Data"

frames = []
for i in tqdm(os.listdir(path)):
  Index = []
  Label = []
  path_new = os.path.join(path, i)
  for j in os.listdir(path_new):
    Index.append(os.path.join(path_new, j))
    Label.append(i)
    
  data = {'Index': Index, 'Label': Label}
  df = pd.DataFrame(data)
  frames.append(df)

df = pd.concat(frames, ignore_index=True)
#df.to_csv("/content/drive/MyDrive/Audio Signal Processing/Data/instruments.csv")