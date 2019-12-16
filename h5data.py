import torch.utils.data as data
import h5py
import math
import numpy as np
import os

class H5DataPrepare(data.Dataset):
    def __init__(self, path):
        self.path_haze = path
        self.length = 0
        self.image_paths_haze = sorted([f for f in os.listdir(self.path_haze)])
        self.length = len(self.image_paths_haze)

    def __getitem__(self, idx):
        f = h5py.File(self.path_haze + "/" + self.image_paths_haze[idx], 'r')
        #rollaxis : bring axis in the position at sencond option to the position at last option
        haze = np.rollaxis(f['haze'][:], 2, 0).astype(np.float32)
        image = np.rollaxis(f['gt'][:], 2, 0).astype(np.float32)
        trans = np.rollaxis(f['trans'][:], 2, 0).astype(np.float32)
        atmos = np.rollaxis(f['ato'][:], 2, 0).astype(np.float32)
        C, H, W = haze.shape
        W1 = 0
        W2 = 0
        H1 = 0
        H2 = 0
        if W % 32:
            W1 = math.floor((32 - W % 32) / 2)
            W2 = math.ceil((32 - W % 32) / 2)
        if H % 32:
            H1 = math.floor((32 - H % 32) / 2)
            H2 = math.ceil((32 - H % 32) / 2)
        # hazy = nn.ReflectionPad2d((W1, W2, H1, H2))(hazy)
        # target = nn.ReflectionPad2d((W1, W2, H1, H2))(target)
        hazy = np.pad(haze, ((0, 0), (H1, H2), (W1, W2)), 'reflect')
        target = np.pad(image, ((0, 0), (H1, H2), (W1, W2)), 'reflect')
        return haze, image, (W1, W2, H1, H2)

    def __len__(self):
        return self.length