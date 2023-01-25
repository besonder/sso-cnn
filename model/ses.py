import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from .cayley import StridedConv

def extract_SESLoss(model: nn.Module, scale=2.0) -> Tensor:
    loss_ses = 0
    for _, layer in model.named_modules():
        if hasattr(layer, 'SESLoss'):
            if "Linear" in layer.__class__.__name__ or "1x1" in layer.__class__.__name__:
                loss_ses += scale * layer.SESLoss
            else:
                loss_ses += layer.SESLoss
    return loss_ses

class SESConv(StridedConv, nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, **kwargs):
        if stride != 1 and stride != 2:
            raise Exception("Only 1 or 2 are allowed for stride")
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, bias=bias, **kwargs)
        self.cin = in_channels
        self.cout = out_channels
        self.stride = stride
        self.loss = torch.nn.MSELoss(reduction='mean')
        self.SESLoss = 0
        
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
            ColNorm = torch.norm(Hfft, dim=1, keepdim=True)
            Hfft = Hfft/ColNorm
            RowNorm = torch.norm(Hfft, dim=2, keepdim=True)
            RowLoss = self.loss(RowNorm, torch.ones_like(RowNorm, dtype=RowNorm.dtype)*np.sqrt(self.xcin/self.cout))
            HTH = torch.einsum('ndp, ndq -> npq', Hfft.conj(), Hfft)
            MHTH = torch.triu(torch.ones((self.xcin, self.xcin), dtype=bool), diagonal=1).repeat(b, 1, 1).to(Hfft.device)
            HTHAngle = torch.abs(HTH[MHTH])
            ColAngLoss = self.loss(HTHAngle, torch.zeros_like(HTHAngle, dtype=HTHAngle.dtype))            
            self.SESLoss = self.xcin*RowLoss + ColAngLoss # ses loss
            # self.SESLoss =ColAngLoss # semi loss
        else:
            RowNorm = torch.norm(Hfft, dim=2, keepdim=True)
            Hfft = Hfft/RowNorm
            ColNorm = torch.norm(Hfft, dim=1, keepdim=True)
            ColLoss = self.loss(ColNorm, torch.ones_like(ColNorm, dtype=ColNorm.dtype)*np.sqrt(self.cout/self.xcin))
            HHT = torch.einsum('npc, nqc -> npq', Hfft.conj(), Hfft)
            MHHT = torch.triu(torch.ones((self.cout, self.cout), dtype=bool), diagonal=1).repeat(b, 1, 1).to(Hfft.device)               
            HHTAngle = torch.abs(HHT[MHHT])
            RowAngLoss = self.loss(HHTAngle, torch.zeros_like(HHTAngle, dtype=HHTAngle.dtype))
            self.SESLoss = self.cout*ColLoss + RowAngLoss # ses loss
            # self.SESLoss = RowAngLoss # semi loss
        yfft = (Hfft @ xfft).reshape(n, n // 2 + 1, self.cout, batches)
        y = torch.fft.irfft2(yfft.permute(3, 2, 0, 1), x.shape[2:])
        if self.bias is not None:
            y += self.bias[:, None, None]          
        return y


class SESConv1x1(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size=1, bias=True, **kwargs):
        assert kernel_size == 1
        super().__init__(in_channels, out_channels, kernel_size, bias=bias, **kwargs)
        self.xcin = in_channels
        self.cout = out_channels
        self.loss = torch.nn.MSELoss(reduction='mean')
        self.H = None
        self.SESLoss = 0
        self.register_forward_pre_hook(self.hook_fn)

    def hook_fn(self, module, input):
        H = self.weight.reshape(self.cout, self.xcin)
        if self.cout >= self.xcin:
            ColNorm = torch.norm(H, dim=0, keepdim=True)
            H = H/ColNorm
            RowNorm = torch.norm(H, dim=1, keepdim=True)
            RowLoss = self.loss(RowNorm, torch.ones_like(RowNorm, dtype=RowNorm.dtype)*np.sqrt(self.xcin/self.cout))
            HTH = H.T @ H
            MHTH = torch.triu(torch.ones((self.xcin, self.xcin), dtype=bool), diagonal=1).to(H.device)
            ColAngLoss = self.loss(HTH[MHTH], torch.zeros_like(HTH[MHTH], dtype=HTH[MHTH].dtype))
            self.SESLoss = self.xcin*RowLoss + ColAngLoss # ses loss
            # self.SESLoss =ColAngLoss # semi loss
        else:
            RowNorm = torch.norm(H, dim=1, keepdim=True)
            H = H/RowNorm
            ColNorm = torch.norm(H, dim=0, keepdim=True)
            ColLoss = self.loss(ColNorm, torch.ones_like(ColNorm, dtype=ColNorm.dtype)*np.sqrt(self.cout/self.xcin))
            HHT = H @ H.T
            MHHT = torch.triu(torch.ones((self.cout, self.cout), dtype=bool), diagonal=1).to(H.device)
            RowAngLoss = self.loss(HHT[MHHT], torch.zeros_like(HHT[MHHT], dtype=HHT[MHHT].dtype))
            self.SESLoss = self.cout*ColLoss + RowAngLoss # ses loss
            # self.SESLoss = RowAngLoss # semi loss
        self.H = H[:, :, None, None]
    
    def forward(self, input: Tensor) -> Tensor:
        return F.conv2d(input, self.H if self.training else self.H.detach(), self.bias)


class SESLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, **kwargs): 
        super().__init__(in_features, out_features, bias, **kwargs)
        self.cout = out_features
        self.xcin = in_features
        self.loss = torch.nn.MSELoss(reduction='mean')
        self.H = None
        self.SESLoss = 0
        self.register_forward_pre_hook(self.hook_fn)

    def hook_fn(self, module, input):
        H = self.weight
        if self.cout >= self.xcin:
            ColNorm = torch.norm(H, dim=0, keepdim=True)
            H = H/ColNorm
            RowNorm = torch.norm(H, dim=1, keepdim=True)
            RowLoss = self.loss(RowNorm, torch.ones_like(RowNorm, dtype=RowNorm.dtype)*np.sqrt(self.xcin/self.cout))
            HTH = H.T @ H
            MHTH = torch.triu(torch.ones((self.xcin, self.xcin), dtype=bool), diagonal=1).to(H.device)
            ColAngLoss = self.loss(HTH[MHTH], torch.zeros_like(HTH[MHTH], dtype=HTH[MHTH].dtype))
            self.SESLoss = self.xcin*RowLoss + ColAngLoss # ses loss
            # self.SESLoss =ColAngLoss # semi loss
        else:
            RowNorm = torch.norm(H, dim=1, keepdim=True)
            H = H/RowNorm
            ColNorm = torch.norm(H, dim=0, keepdim=True)
            ColLoss = self.loss(ColNorm, torch.ones_like(ColNorm, dtype=ColNorm.dtype)*np.sqrt(self.cout/self.xcin))
            HHT = H @ H.T
            MHHT = torch.triu(torch.ones((self.cout, self.cout), dtype=bool), diagonal=1).to(H.device)
            RowAngLoss = self.loss(HHT[MHHT], torch.zeros_like(HHT[MHHT], dtype=HHT[MHHT].dtype))
            self.SESLoss = self.cout*ColLoss + RowAngLoss # ses loss
            # self.SESLoss = RowAngLoss # semi loss
        self.H = H
    
    def forward(self, input: Tensor) -> Tensor:
        input = input.flatten(start_dim=1)
        return F.linear(input, self.H if self.training else self.H.detach(), self.bias)

