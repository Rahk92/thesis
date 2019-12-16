import torch.utils.data as data
import numpy as np
from os import listdir
from os.path import join
from PIL import Image, ImageOps
import random
import math

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])

def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    #y, _, _ = img.split()
    return img

def rescale_img(img_in, scale):
    size_in = img_in.size
    new_size_in = tuple([int(x * scale) for x in size_in])
    img_in = img_in.resize(new_size_in, resample=Image.BICUBIC)
    return img_in

def get_patch(img_in, img_tar, patch_size, ix=-1, iy=-1):
    (ih, iw) = img_in.size

    if ix == -1:
        ix = random.randrange(0, iw - patch_size + 1)
    if iy == -1:
        iy = random.randrange(0, ih - patch_size + 1)

    img_in = img_in.crop((iy, ix, iy + patch_size, ix + patch_size))
    img_tar = img_tar.crop((iy, ix, iy + patch_size, ix + patch_size))
    return img_in, img_tar

def augment(img_in, img_tar, rot=True):
    if random.random() < 0.5:
        img_in = ImageOps.flip(img_in)
        img_tar = ImageOps.flip(img_tar)

    if rot:
        if random.random() < 0.5:
            img_in = ImageOps.mirror(img_in)
            img_tar = ImageOps.mirror(img_tar)

        if random.random() < 0.5:
            img_in = img_in.rotate(180)
            img_tar = img_tar.rotate(180)
    return img_in, img_tar

class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir, target_dir, patch_size, data_augmentation):
        super(DatasetFromFolder, self).__init__()
        self.image_filenames = [join(image_dir, x) for x in listdir(image_dir) if is_image_file(x)]
        self.target_filenames = [join(target_dir, x) for x in listdir(target_dir) if is_image_file(x)]
        self.patch_size = patch_size
        self.data_augmentation = data_augmentation

    def __getitem__(self, index):
        name = self.image_filenames[index]
        input = load_img(self.image_filenames[index])
        target = load_img(self.target_filenames[math.floor(index / 6)])
        input, target = get_patch(input, target, self.patch_size)

        if self.data_augmentation:
            input, target = augment(input, target)
        input = np.array(input).astype(np.float32) / 255
        target = np.array(target).astype(np.float32) / 255
        input = np.rollaxis(input, 2, 0)
        target = np.rollaxis(target, 2, 0)
        return name, input, target

    def __len__(self):
        return len(self.image_filenames)