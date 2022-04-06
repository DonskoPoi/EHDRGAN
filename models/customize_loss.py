import torch
import torch.nn as nn
from basicsr.utils.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register()
class tanhL1(nn.Module):
    def __init__(self):
        super(tanhL1, self).__init__()
    def forward(self, x, y):
        loss = torch.mean(torch.abs(torch.tanh(x) - torch.tanh(y)))
        return loss


@LOSS_REGISTRY.register()
class tanhL2(nn.Module):
    def __init__(self):
        super(tanhL2, self).__init__()
    def forward(self, x, y):
        loss = torch.mean(torch.pow((torch.tanh(x) - torch.tanh(y)), 2))
        return loss