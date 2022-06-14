import torch.nn as nn
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.nn.functional import mse_loss
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
        y_pred, y_theta = y_pred

        grid = F.affine_grid(y_theta, y_true.size())
        y_trans = F.grid_sample(y_true, grid)
        
        # salient = y_trans[y_trans > 0].sum()
        # non_salient = y_trans[y_trans <= 0].sum()
        # salience = salient / (salient + non_salient)

        identity_theta = (torch.eye(2, 3, device='cuda') * 0.5).unsqueeze(0).repeat((len(y_theta), 1, 1))
        theta_loss = DiceLoss.abs_exp_loss(y_theta, identity_theta, 3)

        assert y_pred.size() == y_true.size()
        dscs = torch.zeros(y_pred.shape[1])

        for i in range(y_pred.shape[1]):
          y_pred_ch = y_pred[:, i].contiguous().view(-1)
          y_true_ch = y_trans[:, i].contiguous().view(-1)
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

        return 1. - torch.mean(dscs) + theta_loss
