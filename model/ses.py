import numpy as np
import einops
import torch
import torch.nn as nn
import torch.nn.functional as F


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
        assert self.xcin == self.cout
        super().__init__(self.xcin, self.cout, self.kernel_size, stride=stride, bias=bias)
        downsample = "b c (w k1) (h k2) -> b (c k1 k2) w h"
        if stride == 2:
            self.register_forward_pre_hook(lambda _, x: einops.rearrange(x[0], downsample, k1=2, k2=2))            


def Cayley(W):
    if len(W.shape) == 2:
        return Cayley(W[None])[0]

    _, cin, cin = W.shape
    I = torch.eye(cin, dtype=W.dtype, device=W.device)[None, :, :]
    A = W - W.conj().transpose(1, 2)
    iIpA = torch.inverse(I + A)
    return iIpA @ (I - A)


class CayleyConv2d(StridedConv2d, nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, bias=bias)
        self.register_parameter('alpha', None)
        self.wfft = None
        self.wfft0 = None

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
        wfft = self.shift_matrix * torch.fft.rfft2(self.weight, (n, n)).reshape(self.xcin, self.xcin, n * (n // 2 + 1)).permute(2, 0, 1).conj()

        self.wfft = wfft

        if self.alpha is None:
            self.alpha = nn.Parameter(torch.tensor(wfft.norm().item(), requires_grad=True).to(x.device))
        yfft = (Cayley(self.alpha * wfft / wfft.norm()) @ xfft).reshape(n, n // 2 + 1, self.xcin, batches)
        y = torch.fft.irfft2(yfft.permute(3, 2, 0, 1))
        if self.bias is not None:
            y += self.bias[:, None, None]          
        return y


class SESConv2dF(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=None, stride=1, bias=True, kernel_sequence=None):
        super().__init__()

        if kernel_size is None and kernel_sequence is None:
            raise Exception("kernel_size or kernel_sequence is needed")
        if stride != 1 and stride != 2:
            raise Exception("Only 1 or 2 are allowed for stride")
        if kernel_size is not None:
            assert kernel_sequence is None
            if stride == 1:
                self.kseq = (1, 1, kernel_size)
            else :
                self.kseq = (kernel_size, 1, 1)
        else:
            self.kseq = kernel_sequence

        self.cin = in_channels
        self.cout = out_channels
        self.stride = stride
        self.bias = bias 

        if stride == 2:
            self.xcin = 4*self.cin
        else:
            self.xcin = self.cin

        if self.cout != self.cin and self.stride == 1:
            self.cayley1 = CayleyConv2d(self.cin, self.xcin, self.kseq[0], stride=self.stride, bias=False)
        self.cayley2 = CayleyConv2d(self.cout, self.cout, self.kseq[2], bias=self.bias)

        self.register_buffer('H', None)
        # self.register_buffer('K', None)

        # self.shift_matrix = None

    def genFreqSES(self, n):
        conv = nn.Conv2d(self.xcin, self.cout, self.kseq[1], bias=False)
        s = (conv.weight.shape[2] - 1) // 2
        shift_matrix = self.fft_shift_matrix(n, -s).reshape(n * n, 1, 1).to(conv.weight.device)
        # self.shift_matrix = shift_matrix 
        optimizer = torch.optim.SGD(conv.parameters(), lr=0.01)
        loss = torch.nn.MSELoss(reduction='mean')
        for i in range(150):
            Hfft = shift_matrix*torch.fft.fft2(conv.weight, (n, n)).reshape(self.cout, self.xcin, n * n).permute(2, 0, 1).conj()
            b, _, _ = Hfft.shape
            RowNorm = torch.norm(Hfft, dim=2)

            RowLoss = loss(RowNorm, torch.ones_like(RowNorm, dtype=RowNorm.dtype)*np.sqrt(self.xcin/self.cout))

            ColNorm = torch.norm(Hfft, dim=1)
            ColLoss = loss(ColNorm, torch.ones_like(ColNorm, dtype=ColNorm.dtype))

            if self.cout >= self.xcin:
                HHT = torch.einsum('npc, nqc -> npq', Hfft.conj(), Hfft)
                M = torch.triu(torch.ones((self.cout, self.cout), dtype=bool), diagonal=1).repeat(b, 1, 1).to(Hfft.device)               
                HHTAngle = torch.abs(HHT[M])
                RowAngLoss = loss(HHTAngle, torch.zeros_like(HHTAngle, dtype=HHTAngle.dtype))
                L = self.cout*ColLoss + self.xcin*RowLoss + RowAngLoss
            else:
                HTH = torch.einsum('ndp, ndq -> npq', Hfft.conj(), Hfft)
                M = torch.triu(torch.ones((self.xcin, self.xcin), dtype=bool), diagonal=1).repeat(b, 1, 1).to(Hfft.device)
                HTHAngle = torch.abs(HTH[M])
                ColAngLoss = loss(HTHAngle, torch.zeros_like(HTHAngle, dtype=HTHAngle.dtype))
                L = self.cout*ColLoss + self.xcin*RowLoss + ColAngLoss

            optimizer.zero_grad()
            L.backward()
            optimizer.step()

        self.register_buffer("H", conv.weight.to(self.cayley2.weight.device).detach())

    def fft_shift_matrix(self, n, s):
        shift = torch.arange(0, n).repeat((n, 1))
        shift = shift + shift.T
        return torch.exp(1j * 2 * np.pi * s * shift / n)

    def forward(self, x):
        if self.cout != self.cin and self.stride == 1:
            x = self.cayley1(x)
            if self.H == None:
                _, _, n, _ = x.shape
                self.genFreqSES(n)

            # print("cayley1:", torch.norm(x), x.shape)
            x = F.pad(x, pad=(self.kseq[1]//2,)*4, mode='circular')
            x = F.conv2d(x, self.H)
            # print("H:", torch.norm(x), x.shape)
        x = self.cayley2(x)       
        return x


class SESConv2dT(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=None, stride=1, bias=True, kernel_sequence=None):
        super().__init__()

        if kernel_size is None and kernel_sequence is None:
            raise Exception("kernel_size or kernel_sequence is needed")
        if stride != 1 and stride != 2:
            raise Exception("Only 1 or 2 are allowed for stride")
        if kernel_size is not None:
            assert kernel_sequence is None
            if stride == 1:
                self.kseq = (1, 1, kernel_size)
            else :
                self.kseq = (kernel_size, 1, 1)
        else:
            self.kseq = kernel_sequence

        self.cin = in_channels
        self.cout = out_channels
        self.stride = stride
        self.bias = bias 

        assert self.kseq[1] == 1

        if stride == 2:
            self.xcin = 4*self.cin
        else:
            self.xcin = self.cin

        if self.cout != self.cin and self.stride == 1:
            self.cayley1 = CayleyConv2d(self.cin, self.xcin, self.kseq[0], stride=self.stride, bias=False)
        self.cayley2 = CayleyConv2d(self.cout, self.cout, self.kseq[2], bias=self.bias)

        self.register_buffer('H', None)
        # self.register_buffer('K', None)

        # self.shift_matrix = None       

    def genTimeSES(self):
        loss = torch.nn.MSELoss(reduction='mean')
        A = torch.randn(self.cout, self.xcin)
        A.requires_grad_(True)
        lr = 0.01
        for i in range(300):
            H = A.detach()
            H.requires_grad_(True)
            RowNorm = torch.norm(H, dim=1)
            RowLoss = loss(RowNorm, torch.ones_like(RowNorm, dtype=RowNorm.dtype)*np.sqrt(self.xcin/self.cout))
            ColNorm = torch.norm(H, dim=0)
            ColLoss = loss(ColNorm, torch.ones_like(ColNorm, dtype=ColNorm.dtype))
            if self.cout >= self.xcin:
                HHT = H @ H.T
                M = torch.triu(torch.ones((self.cout, self.cout), dtype=bool), diagonal=1).to(H.device)
                RowAngLoss = loss(HHT[M], torch.zeros_like(HHT[M], dtype=HHT[M].dtype))
                L = self.cout*ColLoss + self.xcin*RowLoss + RowAngLoss
            else:
                HTH = H.T @ H
                M = torch.triu(torch.ones((self.xcin, self.xcin), dtype=bool), diagonal=1).to(H.device)
                ColAngLoss = loss(HTH[M], torch.zeros_like(HTH[M], dtype=HTH[M].dtype))
                L = self.cout*ColLoss + self.xcin*RowLoss + ColAngLoss

            L.backward()
            A = H - lr*H.grad
        H = A[:, :, None, None]
        self.register_buffer("H", H.to(self.cayley2.weight.device).detach())


    def fft_shift_matrix(self, n, s):
        shift = torch.arange(0, n).repeat((n, 1))
        shift = shift + shift.T
        return torch.exp(1j * 2 * np.pi * s * shift / n)

    def forward(self, x):
        if self.cout != self.cin and self.stride == 1:
            x = self.cayley1(x)
            if self.H == None:
                self.genTimeSES()
            # print("cayley1:", torch.norm(x), x.shape)
            x = F.pad(x, pad=(self.kseq[1]//2,)*4, mode='circular')
            x = F.conv2d(x, self.H)
            # print("H:", torch.norm(x), x.shape)
        x = self.cayley2(x)       
        return x



class CayleyLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)
        self.alpha = nn.Parameter(torch.ones(1, dtype=torch.float32, requires_grad=True).cuda())
        self.alpha.data = self.weight.norm()

    def forward(self, X):
        if self.training or self.Q is None:
            self.Q = Cayley(self.alpha * self.weight / self.weight.norm())
        return F.linear(X, self.Q if self.training else self.Q.detach(), self.bias)


class SESLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.cout = out_features
        self.xcin = in_features
        self.bias = bias

        if self.cout != self.xcin:
            self.cayleyLinear1 = CayleyLinear(self.xcin, self.xcin, bias=False)
        self.cayleyLinear2 = CayleyLinear(self.cout, self.cout, bias=self.bias)
        self.register_buffer('H', None)

    def genTimeSES(self):
        loss = torch.nn.MSELoss(reduction='mean')
        A = torch.randn(self.cout, self.xcin)
        A.requires_grad_(True)
        lr = 0.005

        for i in range(300):
            H = A.detach()
            H.requires_grad_(True)
            RowNorm = torch.norm(H, dim=1)
            RowLoss = loss(RowNorm, torch.ones_like(RowNorm, dtype=RowNorm.dtype)*np.sqrt(self.xcin/self.cout))
            ColNorm = torch.norm(H, dim=0)
            ColLoss = loss(ColNorm, torch.ones_like(ColNorm, dtype=ColNorm.dtype))
            if self.cout >= self.xcin:
                HHT = H @ H.T
                M = torch.triu(torch.ones((self.cout, self.cout), dtype=bool), diagonal=1).to(H.device)
                RowAngLoss = loss(HHT[M], torch.zeros_like(HHT[M], dtype=HHT[M].dtype))
                L = self.cout*ColLoss + self.xcin*RowLoss + RowAngLoss
            else:
                HTH = H.T @ H
                M = torch.triu(torch.ones((self.xcin, self.xcin), dtype=bool), diagonal=1).to(H.device)
                ColAngLoss = loss(HTH[M], torch.zeros_like(HTH[M], dtype=HTH[M].dtype))
                L = self.cout*ColLoss + self.xcin*RowLoss + ColAngLoss

            L.backward()
            A = H - lr*H.grad
        self.register_buffer("H", H.to(self.cayleyLinear2.weight.device).detach())  

    def forward(self, x):
        if self.cout != self.xcin:
            x = self.cayleyLinear1(x)
            if self.H == None:
                self.genTimeSES()
            x = F.linear(x, self.H)
        x = self.cayleyLinear2(x)       
        return x