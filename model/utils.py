import torch
from torch import Tensor, nn
from torch.nn import functional as F

############################################################
# From: https://github.com/ColinQiyangLi/LConvNet
############################################################

def get_margin_factor(p):
    if p == "inf":
        return 2.0
    return 2.0 ** ((p - 1) / p)

def margin_loss(y_pred: Tensor, y: Tensor, eps: float, p: float, l_constant: float, order=1) -> Tensor:
    margin = eps * get_margin_factor(p) * l_constant
    return F.multi_margin_loss(y_pred, y, margin=margin, p=order)

## Utils
# TODO: max, min / min, max ?
class GroupSort(nn.Module):
    def forward(self, x):
        a, b = x.split(x.size(1) // 2, 1)
        a, b = torch.max(a, b), torch.min(a, b)
        return torch.cat([a, b], dim=1)
    
class ConvexCombo(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = nn.Parameter(torch.Tensor([0.5])) # maybe should be 0.0
        
    def forward(self, x, y):
        s = torch.sigmoid(self.alpha)
        return s * x + (1 - s) * y
    
class Normalize(nn.Module):
    def __init__(self, mu, std):
        super(Normalize, self).__init__()
        self.mu = mu
        self.std = std
        
    def forward(self, x):
        if self.std is not None:
            return (x - self.mu) / self.std
        return (x - self.mu)