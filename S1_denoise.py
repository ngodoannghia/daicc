import warnings
warnings.simplefilter('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

def denoise(df):
    for col in tqdm(df.columns):
        if col not in ['timestamp','block_id']:
            df[col] = np.floor(df[col]*1000) / 1000
    return df

train = pd.read_csv('analysis/sample/data_train_suffle.csv')
train = denoise(train)
train.to_csv('analysis/train_suffle_denoise.csv', index=False)

test = pd.read_csv('analysis/val.csv')
test = denoise(test)
test.to_csv('analysis/val_denoise.csv', index=False)
