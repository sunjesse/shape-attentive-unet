import torch
import numpy as np
import pathlib
import random
import h5py
import os
import pydicom
from torch.utils.data import Dataset

#Our libs
from transforms import apply_mask, ifft2, to_tensor

class Transform:
    def __init__(self, mask_func, resolution, kspace_x=368):
        self.mask_func = mask_func
        self.resolution = resolution
        self.kspace_x = kspace_x

    def __call__(self, img):
        img = to_tensor(img)
        kspace = ifft2(img)
        ss_kspace, mask = apply_mask(kspace, mask_func)
        return img, ss_kspace, mask


class DCM_to_kspace(Dataset):
    def __init__(self, root, mask_func, resolution, dim=640368):
        self.root = root
        self.transform = Transform(mask_func, resolution)
        self.files = self.read_files()
        self.TRAIN_IMG_PATH = os.path.join(root, 'train', 'img')

    def get_file_names(self, root):
        d = []
        txt_file = os.path.join('')
        with open(txt_file, 'r') as f:
            for line in f:
                d.append(line)
        return d

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        filename="mri_img_%03d" % self.files[i]
        full_img_path = os.path.join(self.root, filename)
        img = pydicom.dcmread(filename).pixel_array
        return self.transform(img)

