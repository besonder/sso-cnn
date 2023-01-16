import torch
from torch import Tensor, nn
from torch.nn import functional as F

############################################################
# From: https://github.com/ColinQiyangLi/LConvNet
# Modified
############################################################

def get_margin_factor(p):
    if p == "inf":
        return 2.0
    return 2.0 ** ((p - 1) / p)

def margin_loss(y_pred: Tensor, y: Tensor, eps: float, p: float, l_constant: float, order=1) -> Tensor:
    margin = eps * get_margin_factor(p) * l_constant
    return F.multi_margin_loss(y_pred, y, margin=margin, p=order)

def extract_SESLoss(model, scale=2.0):
    loss = 0
    for _, layer in model.named_modules():
        if hasattr(layer, 'SESLoss'):
            if "Linear" in layer.__class__.__name__ or "1x1" in layer.__class__.__name__:
                loss += scale * layer.SESLoss
            else:
                loss += layer.SESLoss
    return loss


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

class PlainConv(nn.Conv2d):
    def __init__(
        self, in_channels, out_channels, kernel_size=1, stride=1, padding=1, dilation=1, groups=1, 
        bias=True, padding_mode='zeros', device=None, dtype=None
    ) -> None:
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, 
            dilation, groups, bias, padding_mode, device, dtype
        )

class Linear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)
        
    def forward(self, input: Tensor) -> Tensor:
        input = input.flatten(start_dim=1)
        return super().forward(input)