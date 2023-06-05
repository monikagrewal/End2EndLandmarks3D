import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os, glob
import json
import skimage
import pandas as pd

from scipy.ndimage import zoom
from skimage.measure import label, regionprops
from skimage.filters import threshold_otsu

import pdb


def resample(volume, old_spacing, new_spacing, output_size=512):
    scale = np.array(old_spacing) / np.array(new_spacing)
    volume = zoom(volume, scale, order=0)

    siz = volume.shape[1]
    if siz < output_size:
        pad = (output_size - siz)//2
        volume = np.pad(volume, ((0, 0), (pad, pad), (pad, pad)), mode='constant')
    elif siz > output_size:
        crop = (siz - output_size)//2
        volume = volume[:, crop : siz - crop, crop : siz - crop]

    return volume


def get_valid_mask(img):
    if img.max()==img.min():
        valid_mask = np.zeros_like(img)
        return valid_mask

    thresh = threshold_otsu(img)
    valid_mask = img > 0.2*thresh
    valid_mask, num = label(valid_mask, connectivity=2, return_num=True)
    props = regionprops(valid_mask)
    props = sorted(props, key=lambda x: x.area, reverse=True)

    valid_mask[valid_mask != props[0].label] = 0
    valid_mask[valid_mask == props[0].label] = 1

    return valid_mask.astype(np.float64)


class LandmarkDataset(Dataset):
    """Dataset class defining dataloader"""

    def __init__(self, root_dir, initial_transform, deformation, augmentation=None, is_training=True):
        super(Dataset, self).__init__()
        """
        Args:
            data
        """
        self.datainfo = pd.read_csv(root_dir)
        self.datainfo = self.datainfo[self.datainfo.train == is_training]
        self.datainfo = self.datainfo[self.datainfo.discard == 0]
        self.initial_transform = initial_transform
        self.deformation = deformation
        self.augmentation = augmentation


    def __len__(self):
        return len(self.datainfo)


    def __getitem__(self, idx):
        row = self.datainfo.iloc[idx]
        study_path = row.path

        image1 = self.load_volume(study_path)
        valid_mask1 = get_valid_mask(image1)

        image1, valid_mask1, _ = self.initial_transform(image1, valid_mask1)    
        image2, valid_mask2, deformation = self.deformation(image1, valid_mask1)
        
        if self.augmentation is not None:
            image1, _, _ = self.augmentation(image1)
            image2, _, _ = self.augmentation(image2)

        return image1.astype(np.float32), image2.astype(np.float32), deformation.astype(np.float32), valid_mask1.astype(np.float32), valid_mask2.astype(np.float32)


    def load_volume(self, study_path):
        """
        TODO:
        FILL WITH CODE TO READ IMAGE VOLUME FROM A study_path
        """
        return None


def visualize(volume1, volume2, volume3=None, out_dir="./sanity", base_name=0):
    os.makedirs(out_dir, exist_ok=True)
    slices = volume1.shape[0]

    imlist = []
    for i in range(slices):
        if volume3 is None:
            im = np.concatenate([volume1[i], volume2[i]], axis=1)
        else:
            im = np.concatenate([volume1[i], volume2[i], volume3[i]], axis=1)
        imlist.append(im)
        if len(imlist)==4:
            im = np.concatenate(imlist, axis=0)
            skimage.io.imsave(os.path.join(out_dir, "im_{}_{}.jpg".format(base_name, i)), (im*255).astype(np.uint8))
            imlist = []



if __name__ == '__main__':
    from custom_transforms3d import *

    root_dir = '/export/scratch3/grewal/Data/Projects_JPG_data/ThreeD/MODIR_data_train_split/landmark_train_val_info.csv' 
    
    initial_transform = Compose([
        CropDepthwise(p=1.0, crop_size=48, crop_mode='random'),
        CropInplane(p=1.0, crop_size=128, crop_mode='random', border=50),
        ToTensorShape(p=1.0)
        ])
    

    deformation = AnyOf([
                    # RandomScale3D(p=1.0),
                    # RandomRotate3D(p=1.0),
                    RandomElasticTransform3D(p=1.0, sigma=64)
                    ])

    # augmentation = Compose([
    #             RandomBrightness(),
    #             RandomContrast()
    #             ])
    augmentation = None

    dataset = LandmarkDataset(root_dir, initial_transform, deformation, augmentation=augmentation, is_training=True)
    print(len(dataset))

    for i in range(len(dataset)):
        print(i)
        image1, image2, _, valid_mask1, valid_mask2 = dataset[i]
        image3 = np.abs(image1 - image2)
        print(np.max(image3), np.min(image3))
        image3 = (image3 + 1e-3) / (np.max(image3) + 1e-3)

        visualize(image1[0], image2[0], image3[0], base_name=i)
        if i>20:
            break

