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

sys.path.append('..')
import helpers as h

WINDOW_MAX = 500
WINDOW_MIN = 200
# obtained empirically
GLOBAL_PIXEL_MEAN = 0.1

def expand_bbox(bbox, size, padding=32):
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
  if l + w > size[0] - 1:
    w -= l + w - size[0] - 1
    h = w
  if h + t > size[1] - 1:
    h -= h + t - size[1] - 1
    w = h
  
  l = max(0, l)
  t = max(0, t)
  
  return l, t, w, h

def get_bboxes(img):
  count, label = cv.connectedComponents((img * 255).astype(np.uint8))
  
  # [(left, top, width, height)]
  bboxes = []
  for i in range(1, count):
    bbox = cv.boundingRect(np.uint8(label == i))
    bbox = expand_bbox(bbox, img.shape[-2:])
    bboxes.append(bbox)
  return bboxes


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

  # def get_all_bboxes(self):
  #   '''
  #   Returns an array of centers and their file names in the form of
  #   [(tuple, str)], for example:
  #   [([150, 200], 'D0-1.npy'), ([300, 100], 'D0-1.npy'), ...]
  #   '''
  #   bboxes = []
  #   for file_name in self.file_names:
  #     mask_slice = np.load(p.join(self.directory, 'label', file_name))
  #     bboxes_for_file = get_bboxes(mask_slice)
  #     for bbox in bboxes_for_file:
  #       bboxes.append((bbox, file_name))
    
  #   return bboxes

  def __init__(self, folder, hospital_id='D', cropped=False, input_size=128, center_augmentation=False):
    '''
    folder: one of 'train', 'valid', 'test'
    hospital_id: one of 'D', 'K' or 'R'
    '''
    self.directory = p.join('data', folder)
    self.cropped = cropped
    self.input_size = input_size
    self.center_augmentation = center_augmentation

    all_files = h.listdir(p.join(self.directory, 'label'))
    all_files = np.array(all_files)
    all_files.sort()
    all_files = [f for f in all_files if hospital_id in f]
    self.file_names = all_files

    transforms = [] if self.cropped else [A.CenterCrop(512, 512)]
    if folder == 'train':
      transforms.append(AortaDataset.get_augmentation())
    self.transform = A.Compose(transforms, bbox_params=A.BboxParams(format='coco', label_fields=['labels']))
    
  def __len__(self):
    # return 16 # overfit single batch
    return len(self.file_names)

  def get_file_data(self, idx):
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
    bboxes = get_bboxes(mask)

    # viz_img = mask.copy()
    # for (l, t, w, h) in bboxes:
    #   viz_img = cv.rectangle(viz_img, (l, t), (l + w, t + h), 1, 2)
    # plt.imshow(viz_img)
    # plt.show()

    if self.transform is not None:
      transformed = self.transform(image=scan, mask=mask, bboxes=bboxes, labels=[0] * len(bboxes))
      mask = transformed['mask']
      scan = transformed['image']
      bboxes = np.array(transformed['bboxes'], dtype=np.int)

    # viz_img = mask.copy()
    # for (l, t, w, h) in bboxes:
    #   viz_img = cv.rectangle(viz_img, (l, t), (l + w, t + h), 1, 2)
    # plt.imshow(viz_img)
    # plt.show()

    volume_tensor = torch.from_numpy(scan).unsqueeze(0).float()
    mask_tensor = torch.from_numpy(mask).unsqueeze(0).float()

    return volume_tensor, mask_tensor, bboxes

  def get_crops_for_proposal(self, volume, mask, proposal, bboxes=None):
    bboxes = get_bboxes(proposal) if bboxes is None else bboxes

    crops = []
    for bbox in bboxes:
      l, t, w, h = bbox
      volume_cropped = T.crop(volume, t, l, w, h)
      mask_cropped = T.crop(mask, t, l, w, h)
      crops.append((volume_cropped, mask_cropped, bbox))
    
    return crops

  def get_crops(self, idx):
    volume, mask, bboxes = self.get_file_data(idx)
    crops = self.get_crops_for_proposal(volume, mask, mask, bboxes)
    return crops

  def __getitem__(self, idx):
    if self.cropped:
      crops = get_crops(idx)
      volume_tensor, mask_tensor, _ = crops[np.random.randint(0, len(crops))]
    else:
      volume_tensor, mask_tensor, _ = self.get_file_data(idx)
    
    volume_tensor = F.interpolate(volume_tensor.unsqueeze(0), self.input_size)[0]
    mask_tensor = F.interpolate(mask_tensor.unsqueeze(0), self.input_size)[0]
    return volume_tensor, mask_tensor
