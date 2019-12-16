import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from os.path import join
from os import listdir
import numpy as np
import random
import math

def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    return img


class FolderDataset(data.Dataset):
    def __init__(self, image_dir, target_dir, transforms=None, valid=None):
        super(FolderDataset, self).__init__()
        self.hazy_filenames = [join(image_dir, x) for x in listdir(image_dir)]
        self.target_filenames = [join(target_dir, x) for x in listdir(target_dir)]
        self.transforms = transforms
        self.valid = valid
        self.angles = [-90, 0, 90, 180]

    def __getitem__(self, idx):
        name = self.hazy_filenames[idx]
        hazy = load_img(self.hazy_filenames[idx])
        target = load_img(self.target_filenames[idx])
        '''
        i, j, h, w = transforms.RandomCrop().get_params(hazy, output_size=(512, 512))
        hazy = TF.crop(hazy, i, j, h, w)
        target = TF.crop(target, i, j, h, w)
        '''
        if self.transforms is not None:
            angle = random.choice(self.angles)
            TF.rotate(hazy, angle)
            TF.rotate(target, angle)
            seed = np.random.randint(2147483647)
            random.seed(seed)
            hazy = self.transforms(hazy)
            random.seed(seed)
            target = self.transforms(target)
        elif self.valid is None:
            i, j, h, w = transforms.RandomCrop().get_params(hazy, output_size=(512, 512))
            hazy = TF.crop(hazy, i, j, h, w)
            target = TF.crop(target, i, j, h, w)
            hazy = transforms.ToTensor()(hazy)
            target = transforms.ToTensor()(target)
        elif self.valid is not None:
            oW, oH = hazy.size
            if oH > 2400:
                hazy = transforms.Resize(2400)(hazy)
                target = transforms.Resize(2400)(hazy)
            hazy = transforms.ToTensor()(hazy)
            target = transforms.ToTensor()(target)
            C, H, W = hazy.shape
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
            #hazy = nn.ReflectionPad2d((W1, W2, H1, H2))(hazy)
            #target = nn.ReflectionPad2d((W1, W2, H1, H2))(target)
            hazy = np.pad(hazy, ((0, 0), (H1, H2), (W1, W2)), 'reflect')
            target = np.pad(target, ((0, 0), (H1, H2), (W1, W2)), 'reflect')
        '''
        hazy = transforms.ToPILImage()(hazy)
        target = transforms.ToPILImage()(target)
        hazy.show()
        target.show()
        '''
        return name, hazy, target, (W1, W2, H1, H2)

    def __len__(self):
        length = len(self.target_filenames)
        return length



