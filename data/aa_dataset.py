import sys
import os.path as p

import torch
from torch.utils.data import Dataset
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

import albumentations as A
from torchvision.transforms.functional import center_crop
import torchvision.transforms.functional as T
import torch.nn.functional as F

sys.path.append('..')
import helpers as h

WINDOW_MAX = 500
WINDOW_MIN = 200
# obtained empirically
GLOBAL_PIXEL_MEAN = 0.1

class AortaDataset(Dataset):

  in_channels = 3
  out_channels = 3

  @staticmethod
  def get_augmentation():
    transform = A.Compose([
      A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=.5),
      A.HorizontalFlip(p=0.3),
    ])
    return transform

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
    self.directory = p.join('data', folder)
    self.cropped = cropped
    self.input_size = input_size

    self.transform = None

    all_files = h.listdir(p.join(self.directory, 'label'))
    all_files = np.array(all_files)
    all_files.sort()
    all_files = [f for f in all_files if hospital_id in f]
    self.file_names = all_files

    if folder == 'train':
      self.transform = AortaDataset.get_augmentation()
    else:
      self.transform = None
    
  def __len__(self):
    # return 16 # overfit single batch
    return len(self.file_names)

  def fix_bbox(self, bbox, size, padding=32):
    l, t, w, h = bbox

    # make rect square
    if w < h:
      d = h - w
      l -= d // 2
      w = h
    elif h < w:
      d = w - h
      t -= d // 2
      h = w

    # add padding
    h += padding * 2
    w += padding * 2
    l -= padding
    t -= padding

    # make sure the bbox is witihin image bounds
    if l + w > size - 1:
      w -= l + w - size - 1
      h = w
    if h + t > size - 1:
      h -= h + t - size - 1
      w = h
    
    return l, t, w, h

  def __getitem__(self, idx):
    current_slice_file = self.file_names[idx]

    scan = np.load(p.join(self.directory, 'input', current_slice_file))
    scan = np.expand_dims(scan, axis=0)
    mask = np.load(p.join(self.directory, 'label', current_slice_file))
    mask = np.expand_dims(mask, axis=0)

    # window input slice
    scan[scan > WINDOW_MAX] = WINDOW_MAX
    scan[scan < WINDOW_MIN] = WINDOW_MIN

    scan = scan.astype(np.float64)
    
    # normalize and zero-center
    scan = (scan - WINDOW_MIN) / (WINDOW_MAX - WINDOW_MIN)
    # zero-centered globally because CT machines are calibrated to have even 
    # intensities across images
    scan -= GLOBAL_PIXEL_MEAN

    if self.transform is not None:
      transformed = self.transform(image=scan, mask=mask)
      mask = transformed['mask']
      scan = transformed['image']

    volume_tensor = torch.from_numpy(scan).float()
    mask_tensor = torch.from_numpy(mask).float()

    volume_tensor = center_crop(volume_tensor, (512, 512))
    mask_tensor = center_crop(mask_tensor, (512, 512))

    if self.cropped:
      bbox = cv.boundingRect(np.uint8(mask_tensor[0].numpy() > 0.05))
      l, t, w, h = self.fix_bbox(bbox, 512)
      volume_tensor = T.crop(volume_tensor, t, l, w, h)
      mask_tensor = T.crop(mask_tensor, t, l, w, h)

    volume_tensor = F.interpolate(volume_tensor.unsqueeze(0), self.input_size)[0]
    mask_tensor = F.interpolate(mask_tensor.unsqueeze(0), self.input_size)[0]

    return volume_tensor, mask_tensor
