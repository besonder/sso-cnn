from torch import nn, tensor

from .backbone import KWLarge, ResNet9, WideResNet
from .cayley import Normalize, CayleyConv, CayleyLinear
from .ed import CayleyConvED, CayleyConvED2
from .utils import margin_loss
from utils.option import Config

def get_model(args: Config) -> nn.Sequential:
    if args.conv == "CayleyConv":
        conv = CayleyConv
    elif args.conv == "CayleyConvED":
        conv = CayleyConvED
    elif args.conv == "CayleyConvED2":
        conv = CayleyConvED2

    if args.linear == "CayleyLinear":
        linear = CayleyLinear

    if args.backbone == "KWLarge":
        backbone = KWLarge(conv=conv, linear=linear)
    elif args.backbone == "ResNet9":
        backbone = ResNet9(conv=conv, linear=linear)
    elif args.backbone == "WideResNet":
        backbone = WideResNet(conv=conv, linear=linear)

    mu = tensor((0.4914, 0.4822, 0.4465)).view(3,1,1).cuda()
    std = tensor((0.2471, 0.2435, 0.2616)).view(3,1,1).cuda()

    model = nn.Sequential(
        Normalize(mu, std if args.stddev else 1.0),
        backbone,
    ).cuda()

    return model
