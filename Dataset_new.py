import numpy as np
from torch.utils.data import Dataset, DataLoader
import random
from tqdm import tqdm


class SAKTDataset(Dataset):
    def __init__(self, features, targets, train=True): #HDKIM 100
        super(SAKTDataset, self).__init__()
        self.features = features
        self.targets = targets['anomalous'].values

        self.features = self.features.groupby('breath_id').agg(list).reset_index()

        self.prepare_data()

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.inputs[index].astype('float32'),self.targets[index].astype('float32')

    def prepare_data(self):
        sensor_00 = np.array(self.features['sensor_00'].values.tolist())
        sensor_01 = np.array(self.features['sensor_01'].values.tolist())
        sensor_02 = np.array(self.features['sensor_02'].values.tolist())
        sensor_03 = np.array(self.features['sensor_03'].values.tolist())
        sensor_04 = np.array(self.features['sensor_04'].values.tolist())
        sensor_05 = np.array(self.features['sensor_05'].values.tolist())
        sensor_06 = np.array(self.features['sensor_06'].values.tolist())
        sensor_07 = np.array(self.features['sensor_07'].values.tolist())
        sensor_08 = np.array(self.features['sensor_08'].values.tolist())
        sensor_09 = np.array(self.features['sensor_09'].values.tolist())
        

        self.inputs = np.concatenate([
            sensor_00[:, None], 
            sensor_01[:, None], 
            sensor_02[:, None], 
            sensor_03[:, None], 
            sensor_04[:, None], 
            sensor_05[:, None], 
            sensor_06[:, None], 
            sensor_07[:, None], 
            sensor_08[:, None], 
            sensor_09[:, None], 
        ], 1)

class TestDataset(Dataset):
    def __init__(self, features): #HDKIM 100
        super(TestDataset, self).__init__()
        self.features = features
        self.prepare_data()

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):

        return self.inputs[index].astype('float32')
    
    def prepare_data(self):
        sensor_00 = np.array(self.features['sensor_00'].values.tolist())
        sensor_01 = np.array(self.features['sensor_01'].values.tolist())
        sensor_02 = np.array(self.features['sensor_02'].values.tolist())
        sensor_03 = np.array(self.features['sensor_03'].values.tolist())
        sensor_04 = np.array(self.features['sensor_04'].values.tolist())
        sensor_05 = np.array(self.features['sensor_05'].values.tolist())
        sensor_06 = np.array(self.features['sensor_06'].values.tolist())
        sensor_07 = np.array(self.features['sensor_07'].values.tolist())
        sensor_08 = np.array(self.features['sensor_08'].values.tolist())
        sensor_09 = np.array(self.features['sensor_09'].values.tolist())
        

        self.inputs = np.concatenate([
            sensor_00[:, None], 
            sensor_01[:, None], 
            sensor_02[:, None], 
            sensor_03[:, None], 
            sensor_04[:, None], 
            sensor_05[:, None], 
            sensor_06[:, None], 
            sensor_07[:, None], 
            sensor_08[:, None], 
            sensor_09[:, None], 
        ], 1)
