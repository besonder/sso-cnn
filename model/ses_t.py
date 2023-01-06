import numpy as np
import torch
from torch import Tensor, nn
from .cayley import StridedConv


def extract_SESLoss(model):
    SESLoss = 0
    for _, layer in model.named_modules():
        if all([key in layer.__class__.__name__ for key in ['SES', 'T']]):
            SESLoss += layer.L
    return SESLoss

class SESConv2dFT(StridedConv, nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True):
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
            RowNorm = torch.norm(Hfft, dim=2, keepdim=True)
            RowLoss = self.loss(RowNorm, torch.ones_like(RowNorm, dtype=RowNorm.dtype)*np.sqrt(self.xcin/self.cout))
            ColNorm = torch.norm(Hfft, dim=1, keepdim=True)
            ColLoss = self.loss(ColNorm, torch.ones_like(ColNorm, dtype=ColNorm.dtype))
            HHT = torch.einsum('npc, nqc -> npq', Hfft.conj(), Hfft)
            MHHT = torch.triu(torch.ones((self.cout, self.cout), dtype=bool), diagonal=1).repeat(b, 1, 1).to(Hfft.device)               
            HHTAngle = torch.abs(HHT[MHHT])
            RowAngLoss = self.loss(HHTAngle, torch.zeros_like(HHTAngle, dtype=HHTAngle.dtype))
            HTH = torch.einsum('ndp, ndq -> npq', Hfft.conj(), Hfft)
            MHTH = torch.triu(torch.ones((self.xcin, self.xcin), dtype=bool), diagonal=1).repeat(b, 1, 1).to(Hfft.device)
            HTHAngle = torch.abs(HTH[MHTH])
            ColAngLoss = self.loss(HTHAngle, torch.zeros_like(HTHAngle, dtype=HTHAngle.dtype))            
            self.L = self.cout*ColLoss + 2*self.xcin*RowLoss + 2*RowAngLoss + ColAngLoss
            # self.L = self.cout*ColLoss + ColAngLoss
            yfft = (Hfft/ColNorm @ xfft).reshape(n, n // 2 + 1, self.cout, batches)
        else:
            RowNorm = torch.norm(Hfft, dim=2, keepdim=True)
            RowLoss = self.loss(RowNorm, torch.ones_like(RowNorm, dtype=RowNorm.dtype))
            ColNorm = torch.norm(Hfft, dim=1, keepdim=True)
            ColLoss = self.loss(ColNorm, torch.ones_like(ColNorm, dtype=ColNorm.dtype)*np.sqrt(self.cout/self.xcin))
            HHT = torch.einsum('npc, nqc -> npq', Hfft.conj(), Hfft)
            MHHT = torch.triu(torch.ones((self.cout, self.cout), dtype=bool), diagonal=1).repeat(b, 1, 1).to(Hfft.device)               
            HHTAngle = torch.abs(HHT[MHHT])
            RowAngLoss = self.loss(HHTAngle, torch.zeros_like(HHTAngle, dtype=HHTAngle.dtype))
            HTH = torch.einsum('ndp, ndq -> npq', Hfft.conj(), Hfft)
            M = torch.triu(torch.ones((self.xcin, self.xcin), dtype=bool), diagonal=1).repeat(b, 1, 1).to(Hfft.device)
            HTHAngle = torch.abs(HTH[M])
            ColAngLoss = self.loss(HTHAngle, torch.zeros_like(HTHAngle, dtype=HTHAngle.dtype))
            self.L = 2*self.cout*ColLoss + self.xcin*RowLoss + 2*ColAngLoss + RowAngLoss  
            # self.L = self.xcin*RowLoss + RowAngLoss
            yfft = (Hfft/RowNorm @ xfft).reshape(n, n // 2 + 1, self.cout, batches)
        # yfft = (Hfft/torch.norm(Hfft, dim=(1, 2), keepdim=True)*np.sqrt(self.xcin) @ xfft).reshape(n, n // 2 + 1, self.cout, batches)
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
            RowNorm = torch.norm(H, dim=1, keepdim=True)
            RowLoss = self.loss(RowNorm, torch.ones_like(RowNorm, dtype=RowNorm.dtype)*np.sqrt(self.xcin/self.cout)*self.scale)
            ColNorm = torch.norm(H, dim=0, keepdim=True)
            ColLoss = self.loss(ColNorm, torch.ones_like(ColNorm, dtype=ColNorm.dtype)*self.scale)
            HHT = H @ H.T
            MHHT = torch.triu(torch.ones((self.cout, self.cout), dtype=bool), diagonal=1).to(H.device)
            RowAngLoss = self.loss(HHT[MHHT], torch.zeros_like(HHT[MHHT], dtype=HHT[MHHT].dtype))
            HTH = H.T @ H
            MHTH = torch.triu(torch.ones((self.xcin, self.xcin), dtype=bool), diagonal=1).to(H.device)
            ColAngLoss = self.loss(HTH[MHTH], torch.zeros_like(HTH[MHTH], dtype=HTH[MHTH].dtype))
            self.L = self.cout*ColLoss + 2*self.xcin*RowLoss + 2*RowAngLoss + ColAngLoss
        else:
            RowNorm = torch.norm(H, dim=1, keepdim=True)
            RowLoss = self.loss(RowNorm, torch.ones_like(RowNorm, dtype=RowNorm.dtype)*self.scale)
            ColNorm = torch.norm(H, dim=0, keepdim=True)
            ColLoss = self.loss(ColNorm, torch.ones_like(ColNorm, dtype=ColNorm.dtype)*np.sqrt(self.cout/self.xcin)*self.scale)
            HHT = H @ H.T
            MHHT = torch.triu(torch.ones((self.cout, self.cout), dtype=bool), diagonal=1).to(H.device)
            RowAngLoss = self.loss(HHT[MHHT], torch.zeros_like(HHT[MHHT], dtype=HHT[MHHT].dtype))
            HTH = H.T @ H
            MHTH = torch.triu(torch.ones((self.xcin, self.xcin), dtype=bool), diagonal=1).to(H.device)
            ColAngLoss = self.loss(HTH[MHTH], torch.zeros_like(HTH[MHTH], dtype=HTH[MHTH].dtype))
            self.L = 2*self.cout*ColLoss + self.xcin*RowLoss + 2*ColAngLoss + RowAngLoss


class SESLinearT(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, scale=0.5): 
        super().__init__(in_features, out_features, bias)
        self.cout = out_features
        self.xcin = in_features
        self.scale = scale
        self.loss = torch.nn.MSELoss(reduction='mean')
        self.L = 0
        self.register_forward_pre_hook(self.hook_fn)

    def hook_fn(self, module, input):
        H = self.weight.reshape(self.cout, self.xcin)
        HNorm = torch.norm(H, dim=(0, 1), keepdim=True)
        if self.cout >= self.xcin:
            RowNorm = torch.norm(H, dim=1, keepdim=True)
            RowLoss = self.loss(RowNorm, torch.ones_like(RowNorm, dtype=RowNorm.dtype)*np.sqrt(self.xcin/self.cout)*self.scale)
            ColNorm = torch.norm(H, dim=0, keepdim=True)
            ColLoss = self.loss(ColNorm, torch.ones_like(ColNorm, dtype=ColNorm.dtype)*self.scale)
            HHT = H @ H.T
            MHHT = torch.triu(torch.ones((self.cout, self.cout), dtype=bool), diagonal=1).to(H.device)
            RowAngLoss = self.loss(HHT[MHHT], torch.zeros_like(HHT[MHHT], dtype=HHT[MHHT].dtype))
            HTH = H.T @ H
            MHTH = torch.triu(torch.ones((self.xcin, self.xcin), dtype=bool), diagonal=1).to(H.device)
            ColAngLoss = self.loss(HTH[MHTH], torch.zeros_like(HTH[MHTH], dtype=HTH[MHTH].dtype))
            self.L = self.cout*ColLoss + 2*self.xcin*RowLoss + 2*RowAngLoss + ColAngLoss
            # self.L = self.cout*ColLoss + ColAngLoss
            with torch.no_grad():
                self.weight = torch.nn.Parameter(self.weight/ColNorm*self.scale)
                # self.weight = torch.nn.Parameter(self.weight/HNorm*np.sqrt(self.xcin)*self.scale)
        else:
            RowNorm = torch.norm(H, dim=1, keepdim=True)
            RowLoss = self.loss(RowNorm, torch.ones_like(RowNorm, dtype=RowNorm.dtype)*self.scale)
            ColNorm = torch.norm(H, dim=0, keepdim=True)
            ColLoss = self.loss(ColNorm, torch.ones_like(ColNorm, dtype=ColNorm.dtype)*np.sqrt(self.cout/self.xcin)*self.scale)
            HHT = H @ H.T
            MHHT = torch.triu(torch.ones((self.cout, self.cout), dtype=bool), diagonal=1).to(H.device)
            RowAngLoss = self.loss(HHT[MHHT], torch.zeros_like(HHT[MHHT], dtype=HHT[MHHT].dtype))
            HTH = H.T @ H
            MHTH = torch.triu(torch.ones((self.xcin, self.xcin), dtype=bool), diagonal=1).to(H.device)
            ColAngLoss = self.loss(HTH[MHTH], torch.zeros_like(HTH[MHTH], dtype=HTH[MHTH].dtype))
            self.L = 2*self.cout*ColLoss + self.xcin*RowLoss + 2*ColAngLoss + RowAngLoss
            # self.L = self.xcin*RowLoss + RowAngLoss
            with torch.no_grad():
                self.weight = torch.nn.Parameter(self.weight/RowNorm*self.scale)
                # self.weight = torch.nn.Parameter(self.weight/HNorm*np.sqrt(self.cout)*self.scale)
    
    def forward(self, input: Tensor) -> Tensor:
        input = input.flatten(start_dim=1)
        return super().forward(input)

