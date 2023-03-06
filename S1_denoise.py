import warnings
warnings.simplefilter('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

def denoise(df):
    for col in tqdm(df.columns):
        if col not in ['timestamp','block_id']:
            df[col] = np.floor(df[col]*100)
    return df

train = pd.read_csv('train.csv')
train = denoise(train)
train.to_csv('data/train_denoise.csv', index=False)

test = pd.read_csv('val.csv')
test = denoise(test)
test.to_csv('data/val_denoise.csv', index=False)
