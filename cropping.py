import numpy as np
import cv2 as cv
import torch

import torchvision.transforms.functional as T
import torch.nn.functional as F
from torch.utils.data import Dataset
import albumentations as A


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

  def get_crops_for_proposal(self, volume, mask, proposal, bboxes=None):
    bboxes = get_bboxes(proposal) if bboxes is None else bboxes

    crops = []
    for bbox in bboxes:
      l, t, w, h = bbox
      volume_cropped = T.crop(volume, t, l, w, h)
      mask_cropped = T.crop(mask, t, l, w, h)
      crops.append((volume_cropped, mask_cropped, bbox))
    
    return crops



  def __init__(self, folder, cropped=False, input_size=128, center_augmentation=False):
    self.cropped = cropped
    self.cropped = cropped
    self.input_size = input_size
    self.center_augmentation = center_augmentation

    transforms = [] if self.cropped else [A.CenterCrop(512, 512)]
    if folder == 'train':
      transforms.append(AortaDataset.get_augmentation())
    self.transform = A.Compose(transforms, bbox_params=A.BboxParams(format='coco', label_fields=['labels']))

  def get_file_data(self, idx):
    scan, mask = self.get_img_mask(idx)
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

