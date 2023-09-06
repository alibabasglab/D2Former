import torch
import torch.nn as nn
import torch.nn.functional as F

def l2_norm(s1, s2):
    #norm = torch.sqrt(torch.sum(s1*s2, 1, keepdim=True))
    #norm = torch.norm(s1*s2, 1, keepdim=True)

    norm = torch.sum(s1*s2, -1, keepdim=True)
    return norm

def si_snr(s1, s2, eps=1e-8):
    #s1 = remove_dc(s1)
    #s2 = remove_dc(s2)
    s1_s2_norm = l2_norm(s1, s2)
    s2_s2_norm = l2_norm(s2, s2)
    s_target =  s1_s2_norm/(s2_s2_norm+eps)*s2
    e_nosie = s1 - s_target
    target_norm = l2_norm(s_target, s_target)
    noise_norm = l2_norm(e_nosie, e_nosie)
    snr = 10*torch.log10((target_norm)/(noise_norm+eps)+eps)
    return torch.mean(snr)

def compute_mask_loss(noisy_spec, clean_spec, cmask):
    b,c,d,t = noisy_spec.size()
    #noisy_spec: torch.Size([1, 2, 201, 321])
    Sr = clean_spec[:, 0, :, :]
    Si = clean_spec[:, 1, ::, :]
    #Y = self.stft(noisy)
    Yr = noisy_spec[:, 0, :, :]
    Yi = noisy_spec[:, 1, :, :]
    Y_pow = Yr**2 + Yi**2
    Y_mag = torch.sqrt(Y_pow)
    gth_mask_r = (Sr*Yr+Si*Yi)/(Y_pow + 1e-8)
    gth_mask_i = (Si*Yr-Sr*Yi)/(Y_pow + 1e-8)
    gth_mask_r[gth_mask_r > 2] = 1
    gth_mask_r[gth_mask_r < -2] = -1
    gth_mask_i[gth_mask_i > 2] = 1
    gth_mask_i[gth_mask_i < -2] = -1

    #print('gth_mask_r: {}'.format(gth_mask_r.size()))
    #print('cmask: {}'.format(cmask.size()))
    cmask = cmask.permute(0, 2, 1, 3)
    mask_loss = F.mse_loss(gth_mask_r, cmask[...,0]) + F.mse_loss(gth_mask_i, cmask[...,1])
    #phase_loss = F.mse_loss(gth_mask_i, cmp_mask_i) * d #[:,self.feat_dim:, :], cmp_mask[:,self.feat_dim:, :]) * d
    #all_loss = amp_loss + phase_loss
    return mask_loss

def kaiming_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)
    if isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)
    if isinstance(m, nn.Conv1d):
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)


def power_compress(x):
    real = x[..., 0]
    imag = x[..., 1]
    spec = torch.complex(real, imag)
    mag = torch.abs(spec)
    phase = torch.angle(spec)
    mag = mag**0.3
    real_compress = mag * torch.cos(phase)
    imag_compress = mag * torch.sin(phase)
    return torch.stack([real_compress, imag_compress], 1)


def power_uncompress(real, imag):
    spec = torch.complex(real, imag)
    mag = torch.abs(spec)
    phase = torch.angle(spec)
    mag = mag**(1./0.3)
    real_compress = mag * torch.cos(phase)
    imag_compress = mag * torch.sin(phase)
    return torch.stack([real_compress, imag_compress], -1)


class LearnableSigmoid(nn.Module):
    def __init__(self, in_features, beta=1):
        super().__init__()
        self.beta = beta
        self.slope = nn.Parameter(torch.ones(in_features))
        self.slope.requiresGrad = True

    def forward(self, x):
        return self.beta * torch.sigmoid(self.slope * x)
