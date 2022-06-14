import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms

import matplotlib.pyplot as plt
import numpy as np

import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_encoder

class SpatialTransformer(nn.Module):
    def __init__(self, task_network, input_size=128):
        super(SpatialTransformer, self).__init__()

        self.input_size = input_size

        self.iters = 0

        self.network_1 = smp.Unet(
          encoder_name='resnet18',
          in_channels=1,
          classes=1)

        self.network_2 = smp.Unet(
          encoder_name='resnet18',
          in_channels=1,
          classes=1)
    
    def forward(self, x):
      # network 1
      features1 = self.network_1.encoder(x) # skip connections
      decoder1_output = self.network_1.decoder(*features1)
      output1 = self.network_1.segmentation_head(decoder1_output)

      # sampling using output1
      sampled = output1

      # network 2
      features2 = self.network_1.encoder(sampled)
      features_all = torch.cat(features1, features2) # skip connections from network 1 + 2
      decoder2_output = self.network_2.decoder(*features_all)
      output2 = self.network_2.segmentation_head(decoder2_output)

      # inverse sample output2
      output2_sampled = output2
      return torch.cat([output1, output2])
