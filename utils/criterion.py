import torch
import torch.nn as nn
import math


def log10(x):
      """Convert a new tensor with the base-10 logarithm of the elements of x. """
      return torch.log(x+1e-6) / math.log(10)
    
class SmoothL1Loss(nn.Module):
    def __init__(self, reduction='mean'):
        super(SmoothL1Loss, self).__init__()
        self.l1loss = torch.nn.SmoothL1Loss(reduction=reduction)

    def forward(self, preds, dsms):
        loss = self.l1loss(preds, dsms)
        return loss
    
class CriterionCrossEntropy(nn.Module):
    def __init__(self, reduction='mean', ignore_index=255):
        super(CriterionCrossEntropy, self).__init__()
        self.ignore_index = ignore_index
        self.criterion = torch.nn.CrossEntropyLoss(reduction=reduction, ignore_index=ignore_index)

    def forward(self, preds, target):

        loss = self.criterion(preds, target)
        
        return loss