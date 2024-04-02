import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss

__all__ = ["MDELoss"]


class MDELoss(nn.Module):
    def __init__(self):
        super(MDELoss, self).__init__()

    def forward(self, pred, true):
        squared_diff = torch.sum(torch.square(true - pred), dim=1)
        distance = torch.sqrt(squared_diff)
        mean_distance = torch.mean(distance)
        return mean_distance
    

