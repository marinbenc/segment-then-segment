import sys
import os.path as p

import torch
from torch.utils.data import Dataset
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

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

class AortaDataset(CropDataset):

  @staticmethod
  def get_patient_names(hospital_id):
    all_files = h.listdir('data/label')
    all_slices = [f.split('-')[0] for f in all_files]
    patient_names = list(set(all_slices))
    patient_names = [s for s in patient_names if hospital_id in s]
    patient_names.sort()
    return patient_names

  def __init__(self, folder, hospital_id='D', cropped=False, input_size=128):
    '''
    folder: one of 'train', 'valid', 'test'
    hospital_id: one of 'D', 'K' or 'R'
    '''
    super().__init__(folder, cropped, input_size)
    self.directory = p.join('data', 'aorta', folder)

    all_files = h.listdir(p.join(self.directory, 'label'))
    all_files = np.array(all_files)
    all_files.sort()
    all_files = [f for f in all_files if hospital_id in f]
    self.file_names = all_files

    transforms = [] if self.cropped else [A.CenterCrop(512, 512)]
    if folder == 'train':
      transforms.append(CropDataset.get_augmentation())
    self.transform = A.Compose(transforms, bbox_params=A.BboxParams(format='coco', label_fields=['labels']))

    
  def __len__(self):
    # return 16 # overfit single batch
    return len(self.file_names)

  def get_img_mask(self, idx):
    current_slice_file = self.file_names[idx]

    scan = np.load(p.join(self.directory, 'input', current_slice_file))
    mask = np.load(p.join(self.directory, 'label', current_slice_file))

    # window input slice
    scan[scan > WINDOW_MAX] = WINDOW_MAX
    scan[scan < WINDOW_MIN] = WINDOW_MIN

    scan = scan.astype(np.float64)
    
    # normalize and zero-center
    scan = (scan - WINDOW_MIN) / (WINDOW_MAX - WINDOW_MIN)
    # zero-centered globally because CT machines are calibrated to have even 
    # intensities across images
    scan -= GLOBAL_PIXEL_MEAN

    return scan, mask
