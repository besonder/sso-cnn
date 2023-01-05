import numpy as np
import einops
import torch
import torch.nn as nn


def extract_SESLoss(model):
    SESLoss = 0
    for _, layer in model.named_modules():
        if all([key in layer.__class__.__name__ for key in ['SES', 'T']]):
            SESLoss += layer.L
    return SESLoss


class StridedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True):
        self.cin = in_channels
        self.cout = out_channels
        
        if stride == 2:
            self.xcin = 4 * in_channels
            self.kernel_size = max(1, kernel_size // 2)
        else:
            self.xcin = in_channels
            self.kernel_size = kernel_size
        super().__init__(self.xcin, self.cout, self.kernel_size, stride=stride, bias=bias)
        downsample = "b c (w k1) (h k2) -> b (c k1 k2) w h"
        if stride == 2:
            self.register_forward_pre_hook(lambda _, x: einops.rearrange(x[0], downsample, k1=2, k2=2))            


class SESConv2dFT(StridedConv2d, nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, bias=True):
        if stride != 1 and stride != 2:
            raise Exception("Only 1 or 2 are allowed for stride")
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, bias=bias)
        self.cin = in_channels
        self.cout = out_channels
        self.stride = stride
        self.loss = torch.nn.MSELoss(reduction='mean')
        self.L = 0
        
        if stride == 2:
            self.xcin = 4*self.cin
        else:
            self.xcin = self.cin

    def fft_shift_matrix(self, n, s):
        shift = torch.arange(0, n).repeat((n, 1))
        shift = shift + shift.T
        return torch.exp(1j * 2 * np.pi * s * shift / n)
    
    def forward(self, x):
        batches, _, n, _ = x.shape
        if not hasattr(self, 'shift_matrix'):
            s = (self.weight.shape[2] - 1) // 2
            self.shift_matrix = self.fft_shift_matrix(n, -s)[:, :(n//2 + 1)].reshape(n * (n // 2 + 1), 1, 1).to(x.device)
        xfft = torch.fft.rfft2(x).permute(2, 3, 1, 0).reshape(n * (n // 2 + 1), self.xcin, batches)
        Hfft = self.shift_matrix * torch.fft.rfft2(self.weight, (n, n)).reshape(self.cout, self.xcin, n * (n // 2 + 1)).permute(2, 0, 1).conj()
        b, _, _ = Hfft.shape
        if self.cout >= self.xcin:
            RowNorm = torch.norm(Hfft, dim=2)
            RowLoss = self.loss(RowNorm, torch.ones_like(RowNorm, dtype=RowNorm.dtype)*np.sqrt(self.xcin/self.cout))
            ColNorm = torch.norm(Hfft, dim=1)
            ColLoss = self.loss(ColNorm, torch.ones_like(ColNorm, dtype=ColNorm.dtype))
            HHT = torch.einsum('npc, nqc -> npq', Hfft.conj(), Hfft)
            M = torch.triu(torch.ones((self.cout, self.cout), dtype=bool), diagonal=1).repeat(b, 1, 1).to(Hfft.device)               
            HHTAngle = torch.abs(HHT[M])
            RowAngLoss = self.loss(HHTAngle, torch.zeros_like(HHTAngle, dtype=HHTAngle.dtype))
            self.L = self.cout*ColLoss + self.xcin*RowLoss + RowAngLoss
        else:
            RowNorm = torch.norm(Hfft, dim=2)
            RowLoss = self.loss(RowNorm, torch.ones_like(RowNorm, dtype=RowNorm.dtype))
            ColNorm = torch.norm(Hfft, dim=1)
            ColLoss = self.loss(ColNorm, torch.ones_like(ColNorm, dtype=ColNorm.dtype)*np.sqrt(self.cout/self.xcin))
            HTH = torch.einsum('ndp, ndq -> npq', Hfft.conj(), Hfft)
            M = torch.triu(torch.ones((self.xcin, self.xcin), dtype=bool), diagonal=1).repeat(b, 1, 1).to(Hfft.device)
            HTHAngle = torch.abs(HTH[M])
            ColAngLoss = self.loss(HTHAngle, torch.zeros_like(HTHAngle, dtype=HTHAngle.dtype))
            self.L = self.cout*ColLoss + self.xcin*RowLoss + ColAngLoss        
        yfft = (Hfft @ xfft).reshape(n, n // 2 + 1, self.cout, batches)
        y = torch.fft.irfft2(yfft.permute(3, 2, 0, 1))
        if self.bias is not None:
            y += self.bias[:, None, None]          
        return y


class SESConv2dST1x1(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size=1, bias=True):
        assert kernel_size == 1
        super().__init__(in_channels, out_channels, kernel_size, bias=bias)
        self.xcin = in_channels
        self.cout = out_channels
        self.loss = torch.nn.MSELoss(reduction='mean')
        self.L = 0
        self.register_forward_pre_hook(self.hook_fn)

    def hook_fn(self, module, input):
        H = self.weight.reshape(self.cout, self.xcin)
        if self.cout >= self.xcin:
            RowNorm = torch.norm(H, dim=1)
            RowLoss = self.loss(RowNorm, torch.ones_like(RowNorm, dtype=RowNorm.dtype)*np.sqrt(self.xcin/self.cout))
            ColNorm = torch.norm(H, dim=0)
            ColLoss = self.loss(ColNorm, torch.ones_like(ColNorm, dtype=ColNorm.dtype))
            HHT = H @ H.T
            M = torch.triu(torch.ones((self.cout, self.cout), dtype=bool), diagonal=1).to(H.device)
            RowAngLoss = self.loss(HHT[M], torch.zeros_like(HHT[M], dtype=HHT[M].dtype))
            self.L = self.cout*ColLoss + self.xcin*RowLoss + RowAngLoss
        else:
            RowNorm = torch.norm(H, dim=1)
            RowLoss = self.loss(RowNorm, torch.ones_like(RowNorm, dtype=RowNorm.dtype))
            ColNorm = torch.norm(H, dim=0)
            ColLoss = self.loss(ColNorm, torch.ones_like(ColNorm, dtype=ColNorm.dtype)*np.sqrt(self.cout/self.xcin))
            HTH = H.T @ H
            M = torch.triu(torch.ones((self.xcin, self.xcin), dtype=bool), diagonal=1).to(H.device)
            ColAngLoss = self.loss(HTH[M], torch.zeros_like(HTH[M], dtype=HTH[M].dtype))
            self.L = self.cout*ColLoss + self.xcin*RowLoss + ColAngLoss        


class SESLinearT(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, scale=0.6):
        super().__init__(in_features, out_features, bias)
        self.cout = out_features
        self.xcin = in_features
        self.scale = scale
        self.loss = torch.nn.MSELoss(reduction='mean')
        self.L = 0
        self.register_forward_pre_hook(self.hook_fn)

    def hook_fn(self, module, input):
        H = self.weight.reshape(self.cout, self.xcin)
        if self.cout >= self.xcin:
            RowNorm = torch.norm(H, dim=1)
            RowLoss = self.loss(RowNorm, torch.ones_like(RowNorm, dtype=RowNorm.dtype)*np.sqrt(self.xcin/self.cout)*self.scale)
            ColNorm = torch.norm(H, dim=0)
            ColLoss = self.loss(ColNorm, torch.ones_like(ColNorm, dtype=ColNorm.dtype)*self.scale)
            HHT = H @ H.T
            M = torch.triu(torch.ones((self.cout, self.cout), dtype=bool), diagonal=1).to(H.device)
            RowAngLoss = self.loss(HHT[M], torch.zeros_like(HHT[M], dtype=HHT[M].dtype))
            self.L = self.cout*ColLoss/2 + self.xcin*RowLoss + RowAngLoss
        else:
            RowNorm = torch.norm(H, dim=1)
            RowLoss = self.loss(RowNorm, torch.ones_like(RowNorm, dtype=RowNorm.dtype)*self.scale)
            ColNorm = torch.norm(H, dim=0)
            ColLoss = self.loss(ColNorm, torch.ones_like(ColNorm, dtype=ColNorm.dtype)*np.sqrt(self.cout/self.xcin)*self.scale)
            HTH = H.T @ H
            M = torch.triu(torch.ones((self.xcin, self.xcin), dtype=bool), diagonal=1).to(H.device)
            ColAngLoss = self.loss(HTH[M], torch.zeros_like(HTH[M], dtype=HTH[M].dtype))
            self.L = self.cout*ColLoss + self.xcin*RowLoss + ColAngLoss

