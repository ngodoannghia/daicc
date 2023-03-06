import numpy as np
from torch.utils.data import Dataset, DataLoader
import random
from tqdm import tqdm


class SAKTDataset(Dataset):
    def __init__(self, features, targets, train=True): #HDKIM 100
        super(SAKTDataset, self).__init__()
        self.features = features
        self.targets = targets

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index].astype('float32'),self.targets[index].astype('float32')

class TestDataset(Dataset):
    def __init__(self, features): #HDKIM 100
        super(TestDataset, self).__init__()
        self.features = features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):

        return self.features[index].astype('float32')
