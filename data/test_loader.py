import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import os, nibabel
import sys, getopt
import PIL
from PIL import Image
import imageio
import scipy.misc
import numpy as np
import glob
from torch.utils import data
import torch
import random
from .augmentations import ComposeTest, RandomRotate, PaddingCenterCropTest
from skimage import transform

class AC17Test(data.Dataset):

    def __init__(self,
                 root,
                 augmentations=None,
                 img_norm=True,
                 target_size=(256, 256)
                 ):
        self.target_size = target_size
        self.ROOT_PATH = root
        self.img_norm = img_norm
        self.augmentations = augmentations
        self.TRAIN_IMG_PATH = os.path.join(root, 'test')
        self.list = self.read_files()

    def read_files(self):
        d = []
        txt_file = os.path.join('./data/test_series.txt')
        with open(txt_file, 'r') as f:
            for i, line in enumerate(f):
                l = line.split(' ')
                key, val = int(l[0]), int(l[1][:-2])
                d.append([key, val])
        return d

    def __len__(self):
        return 100

    def __getitem__(self, i): # i is index
        filename = "patient%03d_frame%02d" % (self.list[i][0], self.list[i][1])
        full_img_path = os.path.join(self.TRAIN_IMG_PATH, filename)
        img = nibabel.load(full_img_path+".nii")
        pix_dim = img.header.structarr['pixdim'][1]
        img = np.array(img.get_data())
        orig = img

        pre_shape = img.shape[:-1]
        ratio = float(pix_dim/1.5)
        scale_vector = [ratio, ratio, 1]

        img = transform.rescale(img,
                                scale_vector,
                                order=1,
                                preserve_range=True,
                                multichannel=False,
                                mode='constant')
        post_scale = img

        if self.augmentations is not None:
            img = img.transpose(2, 0, 1)
            img_c = np.zeros((img.shape[0], self.target_size[0], self.target_size[1]))

            for z in range(img.shape[0]):
                if img[z].min() > 0:
                    img[z] -= img[z].min()

                img_tmp = self.augmentations(img[z].astype(np.uint32))

                if self.img_norm:
                    mu = img_tmp.mean()
                    sigma = img_tmp.std()
                    img_tmp = (img_tmp - mu) / (sigma+1e-10)
                img_c[z] = img_tmp

            img = img_c.transpose(1,2,0)
        img = self._transform(img)

        if filename.endswith('01'):
            filename = filename[:-7] + 'ED'
        else:
            filename = filename[:-7] + 'ES'

        data_dict = {
            "name": filename,
            "image": img,
            "orig": orig,
            "post_scale": post_scale,
            "scale": scale_vector
        }

        return data_dict

    def _transform(self, img):
        if img.ndim == 3:
            img = np.expand_dims(img, axis=0)
            img = np.concatenate((img, img, img), axis=0)
        img = torch.from_numpy(img).float()
        return img

if __name__ == '__main__':
    DATA_DIR = "/PATH/TO/AC17/DATA"
    augs = ComposeTest([PaddingCenterCropTest(224)])
    ac17 = AC17Test(DATA_DIR, augmentations=augs)
    dloader = torch.utils.data.DataLoader(ac17, batch_size=1)
    for idx, batch in enumerate(dloader):
        img = batch['image']
        print(img.shape, img.max(), img.min(), idx)
