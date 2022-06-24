import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as T
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms

import scipy.ndimage as ndi
import cv2 as cv

import matplotlib.pyplot as plt
import numpy as np

import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder

from unet_plain import UNet

def lerp(a, b, c):
  return c * a + (1 - c) * b

class DoubleUnetSampler(nn.Module):
    def __init__(self, input_size=128):
        super(DoubleUnetSampler, self).__init__()

        self.input_size = input_size

        self.iters = 0

        self.network_1 = UNet('cuda', in_channels=1, out_channels=1, sigmoid_activation=False)
        self.network_2 = UNet('cuda', in_channels=1, out_channels=1, sigmoid_activation=True, additional_skips=True)
    
    def forward(self, x):
      # network 1

      original_size = x.shape[-1]

      x_low = F.interpolate(x, self.input_size)

      output1, skips1 = self.network_1(x_low)

      # sample output1

      output1 = torch.sigmoid(output1)
      output1_np = output1.detach().cpu().numpy()

      # [[x, y, w, h]]
      bounding_rects = [cv.boundingRect(np.uint8(img[0] > 0.05)) for img in output1_np]
      bounding_rects_scaled = bounding_rects.copy()

      for i in range(len(bounding_rects)):
        l, t, w, h = bounding_rects[i]
        # make rect square
        if w < h:
          d = h - w
          l -= d // 2
          w = h
        elif h < w:
          d = w - h
          t -= d // 2
          h = w

        # add padding of 8 px
        h += 16
        w += 16
        l -= 16 // 2
        t -= 16 // 2

        # make sure the bbox is witihin image bounds
        if l + w > self.input_size - 1:
          w -= l + w - self.input_size - 1
          h = w
        if h + t > self.input_size - 1:
          h -= h + t - self.input_size - 1
          w = h

        l = max(l, 0)
        t = max(t, 0)

        if w <= 0 or h <= 0:
          l, t, w, h = 0, 0, self.input_size - 1, self.input_size - 1
        
        bounding_rects[i] = [l, t, w, h]
        
        scaling = original_size / self.input_size
        l, t, w, h = (np.array([l, t, w, h]) * scaling).astype(np.int)
        bounding_rects_scaled[i] = [l, t, w, h]

      self.last_bounding_rects = bounding_rects_scaled
      crops = [T.crop(img, t, l, w, h) for (img, (l, t, w, h)) in zip(x, bounding_rects_scaled)]
      for i in range(len(crops)):
        crops[i] = F.interpolate(crops[i].unsqueeze(0), self.input_size)[0]
      sampled = torch.stack(crops)

      # network 2
      output2 = self.network_2(sampled, skips1)      

      # for i in range(output2.shape[0]):
      #   if output2[i].sum() < 0.1:
      #     print(bounding_rects[i])
      #     plt.imshow(output1_np[i][0])
      #     plt.show()
      #     plt.imshow(output1_np[i][0] > lerp(output1_np[i].mean(), output1_np[i].max(), 0.25))
      #     plt.show()

      # inverse sample output2

      # [(left, right, top, bottom)]
      paddings = [(l, self.input_size - (l + w), t, self.input_size - (t + h)) 
                  for (l, t, w, h) in bounding_rects]
      # first scale it back to the size of the bbox
      #print(bounding_rects)
      orig_size = [F.interpolate(img.unsqueeze(0), rect[-2])[0] for (img, rect) in zip(output2, bounding_rects)]
      # then pad around the bbox to position it as it was in the image originally
      output2_inverse = [F.pad(img, padding) for (img, padding) in zip(orig_size, paddings)]
      output2_inverse = torch.stack(output2_inverse)

      # back to original size
      output1 = F.interpolate(output1, original_size)
      output2_inverse = F.interpolate(output2_inverse, original_size)

      # self.iters += 1
      # if self.iters % 10 == 0:
      #   for i in range(len(x_low)):
      #     plt.imshow(x_low[i][0].detach().cpu().numpy())
      #     plt.show()
      #     plt.imshow(output1_np[i][0])
      #     plt.show()
      #     plt.imshow(output1_np[i][0] > 0.05 * output1_np[0].max())
      #     plt.show()
      #     plt.imshow(sampled[i][0].detach().cpu().numpy())
      #     plt.show()

      return [output1, output2_inverse]
