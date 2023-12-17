import torch
import torch.nn as nn
import torch.nn.functional as F


class BCELoss(nn.Module):
    # hint 4
    def __init__(self, w_other=1.0, w_bonafide=9.0):
        super().__init__()
        self.bonafide = w_bonafide
        self.other = w_other

    def forward(self, scores, target, **batch):
        weight = torch.zeros(scores.size(0), device=scores.device)
        weight[target == 1] = self.bonafide
        weight[target == 0] = self.other
        return F.binary_cross_entropy(F.sigmoid(scores), target.float(), weight=weight)
