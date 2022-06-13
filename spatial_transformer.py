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

class SpatialTransformer(nn.Module):
    def __init__(self, task_network, input_size=128):
        super(SpatialTransformer, self).__init__()

        self.input_size = input_size

        self.iters = 0

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 28 * 28, 128),
            nn.ReLU(True),
            nn.Linear(128, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[-1].weight.data.zero_()
        self.fc_loc[-1].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float) * 0.5)

        self.task_network = task_network

    # Spatial transformer network forward function
    def stn(self, x, x_low):
        xs = self.localization(x_low)
        xs = xs.view(-1, 10 * 28 * 28)
        theta = self.fc_loc(xs)

        theta[:, 1] = 0
        theta[:, 3] = 0

        # theta[:, 0] = theta[:, 0].clone().clamp(0.2, 1.0)
        # theta[:, 4] = theta[:, 4].clone().clamp(0.2, 1.0)

        # theta[:, 2] = theta[:, 2].clone().clamp(-0.5, 0.5)
        # theta[:, 5] = theta[:, 5].clone().clamp(-0.5, 0.5)

        # theta[:] = torch.tensor([0.5, 0, 0, 0, 0.5, 0])

        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        # plt.imshow(x[0].detach().squeeze().cpu().numpy())
        # plt.show()


        return x, theta

    def forward(self, x):
        original_size = x.shape[-1]
        x_low = F.interpolate(x, size=self.input_size)
        x_trans, theta = self.stn(x, x_low)
        x_trans_low = F.interpolate(x_trans, size=self.input_size)

        x = self.task_network(x_trans_low)

        x = F.interpolate(x, size=original_size)

        last_row = torch.Tensor([[[0, 0, 1]]] * len(theta)).cuda()
        theta_sq = torch.cat([theta, last_row], 1)
        theta_inv = torch.linalg.inv(theta_sq)[:, :2]
        grid = F.affine_grid(theta_inv, x.size())
        x = F.grid_sample(x, grid)

        if self.iters % 1000 == 0:

            print(theta)

            plt.imshow(x_low.squeeze().detach().cpu().numpy()[0])
            plt.show()

            plt.imshow(x_trans_low.squeeze().detach().cpu().numpy()[0])
            plt.show()
        
        self.iters += 1
        return x, theta
