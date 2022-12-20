import numpy as np

import torch
from torch import nn
from .cayley import StridedConv, cayley

class CayleyConvED(StridedConv, nn.Conv2d):
    def __init__(self, *args, **kwargs):
        # super().__init__(*args, **kwargs)
        if 'stride' in kwargs and kwargs['stride'] == 2:
            self.stride = 2
            self.xcin = 4*args[0]
            self.padding = 1
            self.k = max(1, args[2]//2)
        else:
            self.stride = 1
            self.xcin = args[0]
            self.padding = 0
            self.k =args[2]
        super().__init__(args[0], self.xcin, self.k, padding=self.padding, stride=self.stride)
        self.bias = nn.Parameter(torch.zeros(args[1]))
        self.args = args
        self.register_parameter('alpha', None)
        self.H = None


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
        self.H = H.to(self.weight.device).detach()


    def fft_shift_matrix(self, n, s):
        shift = torch.arange(0, n).repeat((n, 1))
        shift = shift + shift.T
        return torch.exp(1j * 2 * np.pi * s * shift / n)

    
    def forward(self, x):
        cout = self.args[1]

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

        cwxfft = self.H @ cayley(self.alpha * wfft / wfft.norm(), ED=True) @ xfft

        yfft = (cwxfft).reshape(n, n // 2 + 1, cout, batches)

        y = torch.fft.irfft2(yfft.permute(3, 2, 0, 1))
        if self.bias is not None:
            y += self.bias[:, None, None]
        return y


class CayleyConvED2(StridedConv, nn.Conv2d):
    def __init__(self, *args, **kwargs):
        # super().__init__(*args, **kwargs)
        if 'stride' in kwargs and kwargs['stride'] == 2:
            self.stride = 2
            self.xcin = 4*args[0]
            self.padding = 1
            self.k = max(1, args[2]//2)
        else:
            self.stride = 1
            self.xcin = args[0]
            self.padding = 0
            self.k =args[2]
        super().__init__(args[0], self.xcin, self.k, padding=self.padding, stride=self.stride)
        self.bias = nn.Parameter(torch.zeros(args[1]))
        self.args = args
        self.register_parameter('alpha', None)
        self.H = None


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
        self.H = H.detach()


    def fft_shift_matrix(self, n, s):
        shift = torch.arange(0, n).repeat((n, 1))
        shift = shift + shift.T
        return torch.exp(1j * 2 * np.pi * s * shift / n)

    
    def forward(self, x):
        cout = self.args[1]

        batches, _, n, _ = x.shape
        if not hasattr(self, 'shift_matrix'):
            s = (self.weight.shape[2] - 1) // 2
            self.shift_matrix = self.fft_shift_matrix(n, -s)[:, :(n//2 + 1)].reshape(n * (n // 2 + 1), 1, 1).to(x.device)

        xfft = torch.fft.rfft2(x).permute(2, 3, 1, 0).reshape(n * (n // 2 + 1), self.xcin, batches)

        wfft = self.shift_matrix * torch.fft.rfft2(self.weight, (n, n)).reshape(self.xcin, self.xcin, n * (n // 2 + 1)).permute(2, 0, 1).conj()
        if self.alpha is None:
            self.alpha = nn.Parameter(torch.tensor(wfft.norm().item(), requires_grad=True).to(x.device))

        wxfft = cayley(self.alpha * wfft / wfft.norm(), ED=True) @ xfft
        yfft = wxfft.reshape(n, n // 2 + 1, self.xcin, batches)
        y = torch.fft.irfft2(yfft.permute(3, 2, 0, 1))
        if self.H == None:
            self.genH(cout, self.xcin)
        y = torch.nn.functional.conv2d(y, self.H.to(x.device))
        if self.bias is not None:
            y += self.bias[:, None, None]
        return y