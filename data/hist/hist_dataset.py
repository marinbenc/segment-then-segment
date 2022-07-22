import sys
import os.path as p

import torch
from torch.utils.data import Dataset
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import scipy.io

from scipy.ndimage.measurements import label as skimage_label
from skimage.measure import regionprops

import albumentations as A
from torchvision.transforms.functional import center_crop
import torchvision.transforms.functional as T
import torch.nn.functional as F

sys.path.append('../..')
import helpers as h
from cropping import CropDataset, get_bboxes

WINDOW_MAX = 500
WINDOW_MIN = 200
# obtained empirically
GLOBAL_PIXEL_MEAN = 0.1

class HistDataset(CropDataset):
  in_channels = 3
  out_channels = 1

  @staticmethod
  def get_augmentation():
    transform = A.Compose([
      A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=.5),
      A.HorizontalFlip(p=0.3),
    ], bbox_params=A.BboxParams(format='coco', label_fields=['labels'], min_area=10))
    return transform

  def __init__(self, folder, cropped=False, input_size=64):
    '''
    folder: one of 'train', 'valid', 'test'
    '''
    super().__init__(folder, cropped, input_size)
    self.directory = p.join('data', 'hist', folder)

    all_files = h.listdir(p.join(self.directory, 'label'))
    all_files = np.array(all_files)
    all_files.sort()
    self.file_names = all_files
    
    self.transform = HistDataset.get_augmentation() if folder == 'train' else None

  def __len__(self):
    # return 16 # overfit single batch
    return len(self.file_names)

  def get_img_mask(self, idx):
    file_name = self.file_names[idx]

    scan = cv.imread(p.join(self.directory, 'input', file_name))
    mask = cv.imread(p.join(self.directory, 'label', file_name), cv.IMREAD_GRAYSCALE)
    mask = mask.astype(np.float)
    mask /= 255.

    return scan, mask
