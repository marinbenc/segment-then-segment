import numpy as np
import cv2 as cv
import torch
import matplotlib.pyplot as plt

import torchvision.transforms.functional as T
import torch.nn.functional as F
from torch.utils.data import Dataset
import albumentations as A


def expand_bbox(bbox, size, padding=32):
  l, t, w, h = bbox

  # make rect square
  if w < h:
    d = h - w
    l -= d // 2 # move box to the left by half of d
    w = h # increase width by d
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
  if l + w > size[1]:
    l -= (l + w) - (size[1])
  if h + t > size[0]:
    t -= (h + t) - (size[0])
  
  l = max(0, l)
  t = max(0, t)
  
  return l, t, w, h

def get_bboxes(img, padding=32):
  count, label = cv.connectedComponents((img * 255).astype(np.uint8))
  
  # [(left, top, width, height)]
  bboxes = []
  for i in range(1, count):
    bbox = cv.boundingRect(np.uint8(label == i))
    bbox = expand_bbox(bbox, img.shape[-2:], padding)
    bboxes.append(bbox)
  return bboxes

class CropDataset(Dataset):
  in_channels = 3
  out_channels = 3

  @staticmethod
  def get_augmentation():
    transform = A.Compose([
      A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=.5),
      A.HorizontalFlip(p=0.3),
    ])
    return transform

  def __init__(self, folder, cropped=False, input_size=128, center_augmentation=False, padding=32):
    self.cropped = cropped
    self.cropped = cropped
    self.input_size = input_size
    self.center_augmentation = center_augmentation
    self.padding = padding

  def get_file_data(self, idx):
    scan, mask = self.get_img_mask(idx)
    bboxes = get_bboxes(mask, self.padding)

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

    if len(scan.shape) == 3:
      volume_tensor = torch.from_numpy(scan).permute(2, 0, 1).float()
    else:
      volume_tensor = torch.from_numpy(scan).unsqueeze(0).float()
    
    mask_tensor = torch.from_numpy(mask).unsqueeze(0).float()

    return volume_tensor, mask_tensor, bboxes

  def get_crops_for_proposal(self, volume, mask, proposal, bboxes=None):
    bboxes = get_bboxes(proposal, self.padding) if bboxes is None else bboxes

    crops = []
    for bbox in bboxes:
      l, t, w, h = bbox
      volume_cropped = T.crop(volume, t, l, h, w)
      mask_cropped = T.crop(mask, t, l, h, w)

      if w > 0 and h > 0:
        crops.append((volume_cropped, mask_cropped, bbox))
    
    return crops
  
  def get_crops(self, idx):
    volume, mask, bboxes = self.get_file_data(idx)
    crops = self.get_crops_for_proposal(volume, mask, mask, bboxes)
    return crops

  def __getitem__(self, idx):
    if self.cropped:
      crops = self.get_crops(idx)
      if len(crops) == 0:
        volume_tensor, mask_tensor, _ = self.get_file_data(idx)
      else:
        volume_tensor, mask_tensor, _ = crops[np.random.randint(0, len(crops))]
    else:
      volume_tensor, mask_tensor, _ = self.get_file_data(idx)
    
    volume_tensor = F.interpolate(volume_tensor.unsqueeze(0), self.input_size)[0]
    mask_tensor = F.interpolate(mask_tensor.unsqueeze(0), self.input_size)[0]

    return volume_tensor, mask_tensor

