import numpy as np

import torch
from torch import Tensor, nn
from .cayley import StridedConv

def cayley_ED(W):
    if len(W.shape) == 2:
        return cayley_ED(W[None])[0]

    _, cin, cin = W.shape
    I = torch.eye(cin, dtype=W.dtype, device=W.device)[None, :, :]
    A = W - W.conj().transpose(1, 2)
    # print((I+A).shape)
    iIpA = torch.inverse(I + A)
    return iIpA @ (I - A)

class CayleyConvED(StridedConv, nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size=1, **kwargs):
        if 'stride' in kwargs and kwargs['stride'] == 2:
            self.stride = 2
            self.xcin = 4*in_channels
            self.padding = 1
            self.k = max(1, kernel_size//2)
        else:
            self.stride = 1
            self.xcin = in_channels
            self.padding = 0
            self.k = kernel_size
        super().__init__(in_channels, self.xcin, self.k, padding=self.padding, stride=self.stride)
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.out_channels = out_channels
        # self.args = args
        self.register_parameter('alpha', None)
        self.register_buffer('H', None)


    def genH(self, n, k, cout, xcin):
        conv = nn.Conv2d(xcin, cout, k, bias=False)
        s = (conv.weight.shape[2] - 1) // 2
        shift_matrix = self.fft_shift_matrix(n, -s)[:, :(n//2 + 1)].reshape(n * (n // 2 + 1), 1, 1).to(conv.weight.device)
        optimizer = torch.optim.SGD(conv.parameters(), lr=0.1) #lr=0.1
        loss = torch.nn.MSELoss(reduction='mean')
        for i in range(100): #iteration 2000
            H = shift_matrix*torch.fft.rfft2(conv.weight, (n, n)).reshape(cout, xcin, n * (n//2+1)).permute(2, 0, 1).conj()
            b, cout, xcin = H.shape
            Hnorm = torch.norm(H, dim=2)
            
            L1 = loss(Hnorm, torch.ones_like(Hnorm, dtype=Hnorm.dtype)*np.sqrt(xcin/cout))
            
            HHnorm = torch.norm(H, dim=1)
            L2 = loss(HHnorm, torch.ones_like(HHnorm, dtype=HHnorm.dtype))

            if cout >= xcin:
                HH = torch.einsum('npc, nqc -> npq', H.conj(), H)
                M = torch.triu(torch.ones((cout, cout), dtype=bool), diagonal=1).repeat(b, 1, 1).to(H.device)
                HHorth = torch.abs(HH[M])
                L3 = loss(HHorth, torch.zeros_like(HHorth, dtype=HHorth.dtype))
                L = (cout-1)/2*L1 + (cout-1)/2*L2 + L3
            else:
                HH = torch.einsum('ndp, ndq -> npq', H.conj(), H)
                M = torch.triu(torch.ones((xcin, xcin), dtype=bool), diagonal=1).repeat(b, 1, 1).to(H.device)               
                HHorth = torch.abs(HH[M])
                L3 = loss(HHorth, torch.zeros_like(HHorth, dtype=HHorth.dtype))
                L = (xcin-1)/2*L1 + (xcin-1)/2*L2 + L3

            optimizer.zero_grad()
            L.backward()
            optimizer.step()

        H = shift_matrix*torch.fft.rfft2(conv.weight, (n, n)).reshape(cout, xcin, n * (n//2+1)).permute(2, 0, 1).conj()
        self.register_buffer("H", H.to(self.weight.device).detach())


    def fft_shift_matrix(self, n, s):
        shift = torch.arange(0, n).repeat((n, 1))
        shift = shift + shift.T
        return torch.exp(1j * 2 * np.pi * s * shift / n)

    
    def forward(self, x: Tensor):
        if len(x.shape) < 4:
            x = x[..., None, None]
        cout = self.out_channels

        batches, _, n, _ = x.shape
        if not hasattr(self, 'shift_matrix'):
            s = (self.weight.shape[2] - 1) // 2
            self.shift_matrix = self.fft_shift_matrix(n, -s)[:, :(n//2 + 1)].reshape(n * (n // 2 + 1), 1, 1).to(x.device)

        xfft = torch.fft.rfft2(x).permute(2, 3, 1, 0).reshape(n * (n // 2 + 1), self.xcin, batches)

        wfft = self.shift_matrix * torch.fft.rfft2(self.weight, (n, n)).reshape(self.xcin, self.xcin, n * (n // 2 + 1)).permute(2, 0, 1).conj()
        if self.alpha is None:
            self.alpha = nn.Parameter(torch.tensor(wfft.norm().item(), requires_grad=True).to(x.device))

        if self.H == None:
            self.genH(n, self.kernel_size[0], cout, self.xcin)

        cwxfft = self.H @ cayley_ED(self.alpha * wfft / wfft.norm()) @ xfft

        yfft = (cwxfft).reshape(n, n // 2 + 1, cout, batches)

        y = torch.fft.irfft2(yfft.permute(3, 2, 0, 1), x.shape[2:])
        if self.bias is not None:
            y += self.bias[:, None, None]
        return y


class CayleyConvED2(CayleyConvED):
    def __init__(self, in_channels, out_channels, kernel_size=1, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, **kwargs)

    def genH(self, cout, xcin):
        loss = torch.nn.MSELoss(reduction='mean')
        A = torch.randn(cout, xcin)
        A.requires_grad_(True)
        lr = 0.1
        for i in range(100):
            H = A.detach()
            H.requires_grad_(True)
            Hnorm = torch.norm(H, dim=1)
            L1 = loss(Hnorm, torch.ones_like(Hnorm, dtype=Hnorm.dtype)*np.sqrt(xcin/cout))
            HHnorm = torch.norm(H, dim=0)
            L2 = loss(HHnorm, torch.ones_like(HHnorm, dtype=HHnorm.dtype))
            if cout >= xcin:
                HH =  H @ H.T
                M = torch.triu(torch.ones((cout, cout), dtype=bool), diagonal=1).to(H.device)
                # HHorth = torch.abs(HH[M])
                L3 = loss(HH[M], torch.zeros_like(HH[M], dtype=HH[M].dtype))
                L = (cout-1)/2*L1 + (cout-1)/2*L2 + L3
            else:
                HH =H.T @ H
                M = torch.triu(torch.ones((xcin, xcin), dtype=bool), diagonal=1).repeat(1, 1).to(H.device)               
                # HHorth = torch.abs(HH[M])
                L3 = loss(HH[M], torch.zeros_like(HH[M], dtype=HH[M].dtype))
                L = (xcin-1)/2*L1 + (xcin-1)/2*L2 + L3
            L.backward()
            A = H - lr*H.grad
        H = A[:, :, None, None]
        self.register_buffer("H", H.to(self.weight.device).detach())


    def fft_shift_matrix(self, n, s):
        shift = torch.arange(0, n).repeat((n, 1))
        shift = shift + shift.T
        return torch.exp(1j * 2 * np.pi * s * shift / n)

    
    def forward(self, x: Tensor):
        if len(x.shape) < 4:
            x = x[..., None, None]
        cout = self.out_channels

        batches, _, n, _ = x.shape
        if not hasattr(self, 'shift_matrix'):
            s = (self.weight.shape[2] - 1) // 2
            self.shift_matrix = self.fft_shift_matrix(n, -s)[:, :(n//2 + 1)].reshape(n * (n // 2 + 1), 1, 1).to(x.device)

        xfft = torch.fft.rfft2(x).permute(2, 3, 1, 0).reshape(n * (n // 2 + 1), self.xcin, batches)

        wfft = self.shift_matrix * torch.fft.rfft2(self.weight, (n, n)).reshape(self.xcin, self.xcin, n * (n // 2 + 1)).permute(2, 0, 1).conj()
        if self.alpha is None:
            self.alpha = nn.Parameter(torch.tensor(wfft.norm().item(), requires_grad=True).to(x.device))

        wxfft = cayley_ED(self.alpha * wfft / wfft.norm()) @ xfft
        yfft = wxfft.reshape(n, n // 2 + 1, self.xcin, batches)
        y = torch.fft.irfft2(yfft.permute(3, 2, 0, 1), x.shape[2:])
        if self.H == None:
            self.genH(cout, self.xcin)
        y = torch.nn.functional.conv2d(y, self.H.to(x.device))
        if self.bias is not None:
            y += self.bias[:, None, None]
        return y