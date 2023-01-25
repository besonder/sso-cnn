import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn, optim
from .cayley import StridedConv


def extract_SESLoss(model: nn.Module, scale=2.0) -> Tensor:
    total_loss_ses = 0
    norm_loss = nn.MSELoss(reduction='mean')
    for m in model.modules():
        class_name = m.__class__.__name__
        if 'SES' in class_name:
            if "Conv" in class_name:
                cout, xcin, _, _ = m.weight.shape
                Hfft = m.shift_matrix * torch.fft.rfft2(m.weight, (m.n, m.n)).reshape(cout, xcin, m.n * (m.n // 2 + 1)).permute(2, 0, 1).conj()
                b, _, _ = Hfft.shape
                if cout >= xcin:
                    ColNorm = torch.norm(Hfft, dim=1, keepdim=True)
                    Hfft = Hfft/ColNorm
                    RowNorm = torch.norm(Hfft, dim=2, keepdim=True)
                    RowLoss = norm_loss(RowNorm, torch.ones_like(RowNorm, dtype=RowNorm.dtype)*np.sqrt(xcin/cout))
                    HTH = torch.einsum('ndp, ndq -> npq', Hfft.conj(), Hfft)
                    MHTH = torch.triu(torch.ones((xcin, xcin), dtype=bool), diagonal=1).repeat(b, 1, 1).to(Hfft.device)
                    HTHAngle = torch.abs(HTH[MHTH])
                    ColAngLoss = norm_loss(HTHAngle, torch.zeros_like(HTHAngle, dtype=HTHAngle.dtype))            
                    loss_ses = xcin*RowLoss + ColAngLoss # ses loss
                    # loss_ses = ColAngLoss # semi loss
                else:
                    RowNorm = torch.norm(Hfft, dim=2, keepdim=True)
                    Hfft = Hfft/RowNorm
                    ColNorm = torch.norm(Hfft, dim=1, keepdim=True)
                    ColLoss = norm_loss(ColNorm, torch.ones_like(ColNorm, dtype=ColNorm.dtype)*np.sqrt(cout/xcin))
                    HHT = torch.einsum('npc, nqc -> npq', Hfft.conj(), Hfft)
                    MHHT = torch.triu(torch.ones((cout, cout), dtype=bool), diagonal=1).repeat(b, 1, 1).to(Hfft.device)               
                    HHTAngle = torch.abs(HHT[MHHT])
                    RowAngLoss = norm_loss(HHTAngle, torch.zeros_like(HHTAngle, dtype=HHTAngle.dtype))
                    loss_ses = cout*ColLoss + RowAngLoss # ses loss
                    # loss_ses = RowAngLoss # semi loss
                total_loss_ses += loss_ses

            elif "Linear" in class_name or "1x1" in class_name:
                H = m.weight
                cout, xcin = m.weight.shape
                if cout >= xcin:
                    ColNorm = torch.norm(H, dim=0, keepdim=True)
                    H = H/ColNorm
                    RowNorm = torch.norm(H, dim=1, keepdim=True)
                    RowLoss = norm_loss(RowNorm, torch.ones_like(RowNorm, dtype=RowNorm.dtype)*np.sqrt(xcin/cout))
                    HTH = H.T @ H
                    MHTH = torch.triu(torch.ones((xcin, xcin), dtype=bool), diagonal=1).to(H.device)
                    ColAngLoss = norm_loss(HTH[MHTH], torch.zeros_like(HTH[MHTH], dtype=HTH[MHTH].dtype))
                    loss_ses = xcin*RowLoss + ColAngLoss # ses loss
                    # loss_ses =ColAngLoss # semi loss
                else:
                    RowNorm = torch.norm(H, dim=1, keepdim=True)
                    H = H/RowNorm
                    ColNorm = torch.norm(H, dim=0, keepdim=True)
                    ColLoss = norm_loss(ColNorm, torch.ones_like(ColNorm, dtype=ColNorm.dtype)*np.sqrt(cout/xcin))
                    HHT = H @ H.T
                    MHHT = torch.triu(torch.ones((cout, cout), dtype=bool), diagonal=1).to(H.device)
                    RowAngLoss = norm_loss(HHT[MHHT], torch.zeros_like(HHT[MHHT], dtype=HHT[MHHT].dtype))
                    loss_ses = cout*ColLoss + RowAngLoss # ses loss
                    # loss_ses = RowAngLoss # semi loss
                total_loss_ses += scale * loss_ses
            else:
                raise NotImplementedError
    return total_loss_ses

def optimize_ses(model: nn.Module, opt: optim.Optimizer, lam=2.0, scale=2.0) -> Tensor:
    loss_ses = extract_SESLoss(model, scale=scale)
    loss_ses = lam * loss_ses
    opt.zero_grad()
    loss_ses.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    opt.step()

    return loss_ses

class SESConv(StridedConv, nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, **kwargs):
        if stride != 1 and stride != 2:
            raise Exception("Only 1 or 2 are allowed for stride")
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, bias=bias, **kwargs)

    def fft_shift_matrix(self, n, s):
        shift = torch.arange(0, n).repeat((n, 1))
        shift = shift + shift.T
        return torch.exp(1j * 2 * np.pi * s * shift / n)
    
    def forward(self, x):
        batches, _, n, _ = x.shape
        cout, xcin, _, _ = self.weight.shape
        if not hasattr(self, 'shift_matrix'):
            s = (self.weight.shape[2] - 1) // 2
            self.shift_matrix = self.fft_shift_matrix(n, -s)[:, :(n//2 + 1)].reshape(n * (n // 2 + 1), 1, 1).to(x.device)
            self.n = n
        xfft = torch.fft.rfft2(x).permute(2, 3, 1, 0).reshape(n * (n // 2 + 1), xcin, batches)
        Hfft = self.shift_matrix * torch.fft.rfft2(self.weight, (n, n)).reshape(cout, xcin, n * (n // 2 + 1)).permute(2, 0, 1).conj()
        if cout >= xcin:
            ColNorm = torch.norm(Hfft, dim=1, keepdim=True)
            Hfft = Hfft/ColNorm
        else:
            RowNorm = torch.norm(Hfft, dim=2, keepdim=True)
            Hfft = Hfft/RowNorm
        yfft = (Hfft @ xfft).reshape(n, n // 2 + 1, cout, batches)
        y = torch.fft.irfft2(yfft.permute(3, 2, 0, 1), x.shape[2:])
        if self.bias is not None:
            y += self.bias[:, None, None]
        return y


class SESLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, **kwargs): 
        super().__init__(in_features, out_features, bias, **kwargs)
    
    def forward(self, input: Tensor) -> Tensor:
        input = input.flatten(start_dim=1)
        H = self.weight
        cout, xcin = self.weight.shape
        if cout >= xcin:
            ColNorm = torch.norm(H, dim=0, keepdim=True)
            H = H/ColNorm
        else:
            RowNorm = torch.norm(H, dim=1, keepdim=True)
            H = H/RowNorm
        return F.linear(input, H if self.training else H.detach(), self.bias)

