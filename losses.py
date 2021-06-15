from functools import reduce

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['MixLoss', 'DiceLoss', 'GHMCLoss', 'FocalLoss']


class MixLoss(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x, y):
        lf, lfw = [], []
        for i, v in enumerate(self.args):
            if i % 2 == 0:
                lf.append(v)
            else:
                lfw.append(v)
        mx = sum([w * l(x, y) for l, w in zip(lf, lfw)])
        return mx


class Dice_loss(nn.Module):
    def __init__(self):
        super(Dice_loss, self).__init__()

    def forward(self, logits, targets):
        num = targets.size(0)
        smooth = 1

        probs = torch.sigmoid(logits)
        m1 = probs.view(-1)
        m2 = targets.view(-1)
        intersection = m1 * m2
        score = 2.0 * (intersection.sum() + smooth) / (m1.sum() + m2.sum() + smooth)
        score = 1 - score

        return score


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=0.3):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, x, y):
        ce = F.binary_cross_entropy_with_logits(x, y)
        fc = self.alpha * (1 - torch.exp(-ce)) ** self.gamma * ce
        return fc