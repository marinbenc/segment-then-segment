import sys
import os.path as p
from math import sqrt


import torch
from torch.utils.data import Dataset
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

import albumentations as A
import torchvision.transforms.functional as F

from aa_dataset import AortaDataset

sys.path.append('..')
import helpers as h

WINDOW_MAX = 500
WINDOW_MIN = 200
# obtained empirically
GLOBAL_PIXEL_MEAN = 0.1

class AortaPatchesDataset(Dataset):
  def __init__(self, backing_dataset, patches_per_image, img_size):
    self.backing_dataset = backing_dataset
    self.patches_per_image = patches_per_image
    self.img_size = img_size

    p_w = int(img_size // sqrt(patches_per_image))
    p_h = int(img_size // sqrt(patches_per_image))

    # [(top, left, height, width)]
    self.bbox_for_patch = []
    for p_x in range(int(sqrt(self.patches_per_image))):
      for p_y in range(int(sqrt(self.patches_per_image))):
        bbox = (p_x * p_h, p_y * p_w, p_h, p_w)
        self.bbox_for_patch.append(bbox)

  def get_viz_img(self, img_idx):
    img, mask = self.backing_dataset[img_idx]
    dataset_idxs = [img_idx * self.patches_per_image + p_idx for p_idx in range(self.patches_per_image)]
    patches_for_image = [self[dataset_idx] for dataset_idx in dataset_idxs]

    viz_img = np.zeros((self.img_size, self.img_size))
    for p_idx, (_, label) in enumerate(patches_for_image):
      top, left, height, width = self.bbox_for_patch[p_idx]
      viz_img = cv.rectangle(viz_img, (left, top), (left + width, top + height), 1 if label else 0.75, 2 if label else 1)
    
    return img, viz_img

  def __len__(self):
    len(self.backing_dataset) * self.patches_per_image

  def __getitem__(self, idx):
    img_idx = idx // self.patches_per_image
    patch_idx = idx % self.patches_per_image
    img, mask = self.backing_dataset[img_idx]

    top, left, height, width = self.bbox_for_patch[patch_idx]

    crop = F.crop(img, top, left, height, width)

    mask_crop = F.crop(mask, top, left, height, width)
    label = 1 if mask_crop[mask_crop > 0.5].any() else 0
    return crop, label
