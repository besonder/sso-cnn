"""
From https://openreview.net/forum?id=Zr5W2LSRhD
Yu, Tan, et al. "Constructing Orthogonal Convolutions in an Explicit Manner." ICLR 2021.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from einops import rearrange

def transpose_filter(conv_filter):
    conv_filter_T = torch.transpose(conv_filter, 0, 1)    
    conv_filter_T = torch.flip(conv_filter_T, [2, 3])
    return conv_filter_T

def l2_normalize_batch(tensor,dim=-1, eps=1e-12):
    norm = (torch.sqrt(torch.sum(tensor.abs().float() * (tensor).abs().float(),dim,keepdim=True)))
    norm = norm+eps#max(norm, eps)
    ans = tensor / norm
    return ans

def fantastic_one_batch(conv_filter, num_iters=50, return_vectors=False):
    b,out_ch, in_ch = conv_filter.shape

    u1 = torch.randn((b,1, in_ch), device='cuda', requires_grad=False)
    u1.data = l2_normalize_batch(u1.data,dim=2)

    v1 = torch.randn((b,out_ch,1),device='cuda', requires_grad=False)
    v1.data = l2_normalize_batch(v1.data,dim=1)

    for i in range(num_iters):
        v1.data = l2_normalize_batch((conv_filter.data*u1.data).sum(2, keepdim=True).data,dim=1)
        u1.data = l2_normalize_batch((torch.conj(conv_filter).data*(v1.data)).sum(1, keepdim=True).data,dim=2)

    sigma1 = torch.sum(conv_filter.data*u1.data*torch.conj(v1.data),(1,2), keepdim=True).abs()
  
    if return_vectors:
        return v1,u1
    else:
        return sigma1.abs()

class ECOConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=None, 
                 bias=True, train_terms=5, eval_terms=10, init_iters=50, update_iters=1, 
                 update_freq=200, correction=0.7):
        super(ECOConv, self).__init__()
        assert (stride==1) or (stride==2)
        self.init_iters = init_iters
        self.out_channels = out_channels
        self.in_channels = in_channels*stride*stride
        self.max_channels = max(self.out_channels, self.in_channels)
        
        self.stride = stride
        self.kernel_size = kernel_size
        self.update_iters = update_iters
        self.update_freq = update_freq
        self.total_iters = 0
        self.train_terms = train_terms
        self.eval_terms = eval_terms
        self.idx = [0,1,1,2,3,4,2,4,3]
        self.num = 1

        if kernel_size == 1:
            correction = 1.0
        if kernel_size == 1:
            self.random_conv_filter = nn.Parameter(torch.Tensor(torch.randn(self.max_channels, 
                                               self.max_channels, 1)).cuda(),
                                               requires_grad=True)
        else:
            self.random_conv_filter = nn.Parameter(torch.Tensor(torch.randn(self.max_channels,
                                               self.max_channels, 5)).cuda(),
                                               requires_grad=True)
            self.num = 5
        self.vs = []
        self.us = []
        
        random_conv_filter = self.random_conv_filter.reshape(self.max_channels,self.max_channels,1,1,self.num).half()#[:,:,i]
        random_conv_filter_T = transpose_filter(random_conv_filter)
        conv_filter = (0.5*(random_conv_filter - random_conv_filter_T)).permute(4,0,1,2,3).reshape(self.num,self.max_channels,self.max_channels)

        with torch.no_grad():
            u1, v1 = fantastic_one_batch(conv_filter, num_iters=self.init_iters, return_vectors=True)
            self.us = u1#.append(nn.Parameter(u1, requires_grad=False))
            self.vs = v1#.append(nn.Parameter(v1, requires_grad=False))
            
        self.correction = nn.Parameter(torch.Tensor([correction]).cuda(), requires_grad=False)
            
        self.enable_bias = bias
        if self.enable_bias:
            self.bias = nn.Parameter(
                torch.Tensor(self.out_channels).cuda(), requires_grad=True)
        else:
            self.bias = None
        self.reset_parameters()
    def reset_parameters(self):
        stdv = 1.0 / np.sqrt(self.max_channels)
        nn.init.normal_(self.random_conv_filter, std=stdv)
        
        stdv = 1.0 / np.sqrt(self.out_channels)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -stdv, stdv)
            
    def update_sigma(self):
        if self.training:
            if self.total_iters % self.update_freq == 0:
                update_iters = self.init_iters
            else:
                update_iters = self.update_iters
            self.total_iters = self.total_iters + 1
        else:
            update_iters = 0
        sigmas = []
        random_conv_filter = self.random_conv_filter.reshape(self.max_channels,self.max_channels,1,1,self.num)#[:,:,i]
        random_conv_filter_T = transpose_filter(random_conv_filter)
        conv_filter = (0.5*(random_conv_filter - random_conv_filter_T)).permute(4,0,1,2,3).reshape(self.num,self.max_channels,self.max_channels)
        with torch.no_grad():
            for j in range(update_iters):
                self.vs.data = l2_normalize_batch((conv_filter*self.us).sum(
                                            2, keepdim=True).data,dim=1)
                self.us.data = l2_normalize_batch((conv_filter*self.vs).sum(
                                            1, keepdim=True).data,dim=2)
        sigmas = torch.sum(conv_filter*self.us*self.vs,(1,2))
        return sigmas

    def forward(self, x):
        if len(x.shape) == 2:
            x = x[..., None, None]
        conv_filters = []
        sigmas = self.update_sigma()
        random_conv_filter = self.random_conv_filter.reshape(self.max_channels,self.max_channels,1,1,self.num)#[:,:,i]
        random_conv_filter_T = transpose_filter(random_conv_filter)
        conv_filter = (0.5*(random_conv_filter - random_conv_filter_T))
        conv_filter_n = conv_filter.permute(4,0,1,2,3).reshape(self.num,self.max_channels,self.max_channels)
        conv_filter_n = conv_filter_n.div(sigmas.unsqueeze(1).unsqueeze(1)).half()
        curr_conv = conv_filter_n.clone()
        conv_filter = conv_filter_n.clone()
        if self.training:
            num_terms = self.train_terms
        else:
            num_terms = self.eval_terms
        for i in range(2, num_terms+1):
            curr_conv = curr_conv.bmm(conv_filter_n)/float(i)
            conv_filter = (conv_filter + curr_conv)
        conv_filter = conv_filter + torch.eye(self.max_channels).type_as(conv_filter).unsqueeze(0)
        conv_filter = conv_filter.permute(1,2,0)


        if self.stride > 1:
            x = rearrange(x, "b c (w k1) (h k2) -> b (c k1 k2) w h",
                          k1=self.stride, k2=self.stride)

        if self.out_channels > self.in_channels:
            diff_channels = self.out_channels - self.in_channels
            p4d = (0, 0, 0, 0, 0, diff_channels, 0, 0)
            curr_z = F.pad(x, p4d)
        else:
            curr_z = x
        curr_z = curr_z.half()



        w = curr_z.shape[2]
        padold = math.ceil(w*1.0/self.kernel_size)*self.kernel_size - w
        w = curr_z.shape[2] + padold

        if self.kernel_size > 1:
            kernels = []
            for i in self.idx:
                kernels.append(conv_filter[:,:,i].unsqueeze(-1))
            kernels = torch.cat(kernels,2) 
            conv_filter = kernels.reshape(self.max_channels,self.max_channels,self.kernel_size,self.kernel_size)
            N = self.kernel_size
            z4 = conv_filter
            rep = w//N
            zr = torch.fft.ifft2(z4.float())

            conv_filter = zr.real.reshape(self.max_channels,self.max_channels,N,N).half()
        else:
            conv_filter = (conv_filter).reshape(self.max_channels,self.max_channels,1,1).half()


        curr_fact = 1.
        if self.kernel_size == 1:
            z = F.conv2d(curr_z, conv_filter, padding=(self.kernel_size//2, self.kernel_size//2))
        else:
            pad = w//self.kernel_size
            curr_z_2 = F.pad(curr_z, (pad,pad,pad,pad), mode='circular')
            z = F.conv2d(curr_z_2, conv_filter,dilation=w//self.kernel_size)  

        if self.out_channels < self.in_channels:
            z = z[:, :self.out_channels, :, :]
            
        if self.enable_bias:
            z = z + self.bias.view(1, -1, 1, 1)
        return z
