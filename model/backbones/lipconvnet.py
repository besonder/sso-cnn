"""
From https://openreview.net/forum?id=Zr5W2LSRhD
Yu, Tan, et al. "Constructing Orthogonal Convolutions in an Explicit Manner." ICLR 2021.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..bcop import BCOPConv
from ..soc import SOCConv
from ..eco import ECOConv

conv_mapping = {
    'standard': nn.Conv2d,
    'skew': SOCConv,
    'bcop': BCOPConv,
    'eco': ECOConv,
}

class MinMax(nn.Module):
    def __init__(self):
        super(MinMax, self).__init__()

    def forward(self, z, axis=1):
        a, b = z.split(z.shape[axis] // 2, axis)
        c, d = torch.min(a, b), torch.max(a, b)
        return torch.cat([c, d], dim=axis)

class LipBlock(nn.Module):
    def __init__(self, in_planes, planes, conv_module, stride=1, kernel_size=3):
        super(LipBlock, self).__init__()
        self.activation = MinMax()
        self.conv = conv_module(in_planes, planes*stride, 
                                kernel_size=kernel_size, 
                                stride=stride, padding=1)
    def forward(self, x):
        x = self.activation((self.conv(x)))
        return x

class LipNet(nn.Module):
    def __init__(self, block, num_blocks, conv_module, linear, in_channels=32, num_classes=10, input_spatial_shape=32):
        super(LipNet, self).__init__()
        self.activation = MinMax()
        self.in_planes = 3
        self.layer1 = self._make_layer(block, in_channels, num_blocks[0], conv_module, 
                                       stride=2, kernel_size=3)
        self.layer2 = self._make_layer(block, self.in_planes, num_blocks[1], conv_module, 
                                       stride=2, kernel_size=3)
        self.layer3 = self._make_layer(block, self.in_planes, num_blocks[2], conv_module, 
                                       stride=2, kernel_size=3)
        self.layer4 = self._make_layer(block, self.in_planes, num_blocks[3], conv_module,
                                       stride=2, kernel_size=3)
        self.layer5 = self._make_layer(block, self.in_planes, num_blocks[4], conv_module, 
                                       stride=2, kernel_size=1)
        
        flat_size = input_spatial_shape // 32
        flat_features = flat_size * flat_size * self.in_planes
        self.final_conv = linear(flat_features, num_classes)


    def _make_layer(self, block, planes, num_blocks, conv_module, stride, kernel_size):
        strides = [1]*(num_blocks-1) + [stride]
        kernel_sizes = [3]*(num_blocks-1) + [kernel_size]*1
        layers = []
        for stride, kernel_size in zip(strides, kernel_sizes):
            layers.append(block(self.in_planes, planes, conv_module, stride, kernel_size))
            self.in_planes = planes * stride #* stride
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.final_conv(x)
        x = x.view(x.shape[0], -1)
        return x

def LipNet_n(conv: ECOConv, linear, num_blocks=4, num_classes=10) -> LipNet:
    # conv_module = conv_mapping[conv_module_name]
    in_channels = 32
    input_spatial_shape = 32
    num_blocks_list = [num_blocks]*5
    return LipNet(LipBlock, num_blocks_list, conv, linear, in_channels=in_channels, 
                  num_classes=num_classes, input_spatial_shape=input_spatial_shape)
