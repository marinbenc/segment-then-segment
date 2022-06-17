import torch.nn as nn
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.nn.functional import mse_loss, binary_cross_entropy_with_logits as bce_loss
import torch.nn.functional as F


class DiceLoss(nn.Module):

    def __init__(self):
        super(DiceLoss, self).__init__()
        self.smooth = 1.0
        self.iters = 0

    @staticmethod
    def abs_exp_loss(y_pred, y_true, pow):
        return torch.abs((y_pred - y_true) ** pow).mean()

    def forward(self, y_pred, y_true):
        y_pred_1, y_pred_2 = y_pred

        bce = bce_loss(y_pred_1, y_true, pos_weight=torch.ones(1).cuda() * 2.)
        dscs = torch.zeros(y_pred_2.shape[1])

        for i in range(y_pred_2.shape[1]):
          y_pred_ch = y_pred_2[:, i].contiguous().view(-1)
          y_true_ch = y_true[:, i].contiguous().view(-1)
          intersection = (y_pred_ch * y_true_ch).sum()
          dscs[i] = (2. * intersection + self.smooth) / (
              y_pred_ch.sum() + y_true_ch.sum() + self.smooth
          )

        # self.iters += 1
        # if self.iters % 200 == 0:
        #     pred = y_pred[0].squeeze().detach().cpu().numpy()
        #     true = y_true[0].squeeze().detach().cpu().numpy()
        #     plt.imshow(np.dstack((pred * 255, pred * 255, true * 255)))
        #     plt.show()

        return (1. - torch.mean(dscs)) * 0.5 + bce * 0.5
