from torch import Tensor
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