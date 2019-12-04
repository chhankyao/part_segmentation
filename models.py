import torch
import torch.nn as nn


class PartBasis(nn.Module):
    def __init__(self, dim_feat, k):
        super(PartBasis, self).__init__()
        self.w = nn.Parameter(torch.abs(torch.FloatTensor(dim_feat, k).normal_()))

    def forward(self, x=None):
        out = nn.ReLU()(self.w)
        return out
