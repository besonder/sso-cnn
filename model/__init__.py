import torch
from torch import nn, tensor

from .backbones import KWLarge, ResNet9, WideResNet, LipNet_n
from .bcop import BCOPConv
from .cayley import CayleyConv, CayleyLinear
from .soc import SOCConv
from .eco import ECOConv
from .ed import CayleyConvED, CayleyConvED2
from .ses import SESConv2dF, SESConv2dS, SESLinear
from .ses_t import extract_SESLoss, SESConv2dFT, SESConv2dST1x1, SESLinearT
from .utils import margin_loss, Normalize, PlainConv, Linear
from utils.option import Config

def get_model(args: Config) -> nn.Sequential:
    if args.conv == "CayleyConv":
        conv = CayleyConv
    elif args.conv == "CayleyConvED":
        conv = CayleyConvED
    elif args.conv == "CayleyConvED2":
        conv = CayleyConvED2
    elif args.conv == "SESConv2dF":
        conv = SESConv2dF
    elif args.conv == "SESConv2dS":
        conv = SESConv2dS       
    elif args.conv == "SESConv2dFT":
        conv = SESConv2dFT
    elif args.conv == "SESConv2dST1x1":
        conv = SESConv2dST1x1
    elif args.conv == "ECO":
        conv = ECOConv
    elif args.conv == "BCOP":
        conv = BCOPConv
    elif args.conv == "SOC":
        conv = SOCConv
    elif args.conv == "PlainConv":
        conv = PlainConv

    if args.linear == "Linear":
        linear = Linear
    elif args.linear == "CayleyLinear":
        linear = CayleyLinear
    elif args.linear == 'SESLinear':
        linear = SESLinear
    elif args.linear == "SESLinearT":
        linear = SESLinearT
    else:
        assert args.conv == args.linear, "conv and linear should be same."
        linear = conv
        args.logger("Conv and Linear are same.")

    if args.backbone == "KWLarge":
        backbone = KWLarge(conv=conv, linear=linear, num_classes=args.num_classes)
    elif args.backbone == "ResNet9":
        backbone = ResNet9(conv=conv, linear=linear, num_classes=args.num_classes)
    elif args.backbone == "WideResNet":
        backbone = WideResNet(conv=conv, linear=linear, num_classes=args.num_classes)
    elif "LipConvNet" in args.backbone:
        backbone = LipNet_n(conv=conv, linear=linear, num_blocks=args.n_lip, num_classes=args.num_classes)

    mu = tensor((0.4914, 0.4822, 0.4465)).view(3,1,1).cuda()
    std = tensor((0.2471, 0.2435, 0.2616)).view(3,1,1).cuda()

    model = nn.Sequential(
        Normalize(mu, std),
        backbone,
    ).cuda()

    # model(torch.randn(size=(64, 3, 32, 32)).cuda())

    return model
