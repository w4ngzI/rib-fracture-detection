import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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

if __name__ == '__main__':
    logits = torch.randn((4, 1, 64, 64,64))
    print(logits[:,0,0,0,0])
    targets = torch.ones((4, 1, 64, 64, 64))

    diceloss = Dice_loss()
    loss = diceloss(logits, targets)
    print(loss)