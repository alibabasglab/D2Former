import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange
from einops.layers.torch import Rearrange

# source: https://github.com/lucidrains/conformer/blob/master/conformer/conformer.py
# helper functions


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def calc_same_padding(kernel_size):
    pad = kernel_size // 2
    return (pad, pad - (kernel_size + 1) % 2)

def Csigmoid(x):
    a, b = x[...,0], x[...,1]
    denominator = 1 + 2 * torch.exp(-a) * torch.cos(b) + torch.exp(-2 * a)
    real = 1 + torch.exp(-a) * torch.cos(b) / denominator
    imag = torch.exp(-a) * torch.sin(b) / denominator
    return torch.stack((real, imag), dim=-1)

class CSwish(nn.Module):
    def forward(self, x):
        a, b = x[...,0], x[...,1]
        c = a.sigmoid()
        d = b.sigmoid()
        #y = Csigmoid(x)
        #c, d = y[...,0], y[...,1]
        return torch.stack((a*c-b*d, a*d+b*c), dim=-1)

class CGLU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        out, gate = x.chunk(2, dim=self.dim)
        a, b = out[...,0], out[...,1]
        #gate = Csigmoid(gate)        
        c, d = gate[...,0], gate[...,1]
        c = c.sigmoid()
        d = d.sigmoid()
        #return out * gate.sigmoid()
        return torch.stack((a*c-b*d, a*d+b*c), dim=-1)


class CDepthWiseConv1d(nn.Module):
    def __init__(self, chan_in, chan_out, kernel_size, padding):
        super().__init__()
        self.padding = padding
        self.conv_r = nn.Conv1d(chan_in, chan_out, kernel_size, groups = chan_in)
        self.conv_i = nn.Conv1d(chan_in, chan_out, kernel_size, groups = chan_in)

    def forward(self, x):
        a, b = x[...,0], x[...,1]
        a = F.pad(a, self.padding)
        b = F.pad(b, self.padding)
        return torch.stack((self.conv_r(a)-self.conv_i(b), self.conv_r(b)+self.conv_i(a)), dim=-1)

class CConv1d(nn.Module):
    def __init__(self, chan_in, chan_out, kernel_size):
        super().__init__()
        self.conv_r = nn.Conv1d(chan_in, chan_out, kernel_size)
        self.conv_i = nn.Conv1d(chan_in, chan_out, kernel_size)

    def forward(self, x):
        a, b = x[...,0], x[...,1]
        return torch.stack((self.conv_r(a)-self.conv_i(b), self.conv_r(b)+self.conv_i(a)), dim=-1)
# attention, feedforward, and conv module


class Scale(nn.Module):
    def __init__(self, scale, fn):
        super().__init__()
        self.fn = fn
        self.scale = scale

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) * self.scale


class CPreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm_r = nn.LayerNorm(dim)
        self.norm_i = nn.LayerNorm(dim)

    def forward(self, x, **kwargs):
        a, b = x[...,0], x[...,1]
        a = self.norm_r(a)
        b = self.norm_i(b)
        return self.fn(torch.stack((a,b),dim=-1), **kwargs)

class CLayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm_r = nn.LayerNorm(dim)
        self.norm_i = nn.LayerNorm(dim)

    def forward(self, x):
        a, b = x[...,0], x[...,1]
        a = self.norm_r(a)
        b = self.norm_i(b)
        return torch.stack((a,b),dim=-1)

class CLinear(nn.Module):

    def __init__(self, in_features, out_features, bias=True):
        super(CLinear, self).__init__()
        self.fc_r = nn.Linear(in_features, out_features, bias=bias)
        self.fc_i = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x):
        a, b = x[...,0], x[...,1]
        return torch.stack((self.fc_r(a)-self.fc_i(b), self.fc_r(b)+self.fc_i(a)), dim=-1)

def Csoftmax(x, dim):
    a, b = x[...,0], x[...,1]
    return torch.softmax(torch.sqrt(a**2 + b**2), dim=dim)

class CDropout(nn.Module):
    def __init__(self, prob):
        super().__init__()
        self.dropout = nn.Dropout(prob) 
       
    def forward(self, x):
        a, b = x[...,0:1], x[...,1:]
        c = torch.ones(x[...,0:1].size()).to(a)
        c = self.dropout(c)
        return torch.cat((a*c, b*c), dim=-1)

class CBatchNorm1d(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.bn_r = nn.BatchNorm1d(num_features=num_features)
        self.bn_i = nn.BatchNorm1d(num_features=num_features)

    def forward(self, x):
        a = self.bn_r(x[...,0])
        b = self.bn_i(x[...,1])
        return torch.stack((a, b), dim=-1)

class CAttention(nn.Module):
    def __init__(
        self,
        dim,
        heads = 8,
        dim_head = 64,
        dropout = 0.,
        max_pos_emb = 512
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads= heads
        self.scale = dim_head ** -0.5
        self.to_q = CLinear(dim, inner_dim, bias = False)
        self.to_kv = CLinear(dim, inner_dim * 2, bias = False)
        self.to_out = CLinear(inner_dim, dim)

        self.max_pos_emb = max_pos_emb
        self.rel_pos_emb = nn.Embedding(2 * max_pos_emb + 1, dim_head)

        self.dropout = CDropout(dropout)

    def forward(self, x, context = None, mask = None, context_mask = None):
        n, device, h, max_pos_emb, has_context = x.shape[-3], x.device, self.heads, self.max_pos_emb, exists(context)
        context = default(context, x)

        q, k, v = (self.to_q(x), *self.to_kv(context).chunk(2, dim = -2))
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) c -> b h n d c', h = h), (q, k, v))

        q_r, q_i = q[...,0], q[...,1]
        k_r, k_i = k[...,0], k[...,1]
        dots_r = einsum('b h i d, b h j d -> b h i j', q_r, k_r) * self.scale - einsum('b h i d, b h j d -> b h i j', q_i, k_i) * self.scale
        dots_i = einsum('b h i d, b h j d -> b h i j', q_r, k_i) * self.scale + einsum('b h i d, b h j d -> b h i j', q_i, k_r) * self.scale

        # shaw's relative positional embedding
        seq = torch.arange(n, device = device)
        dist = rearrange(seq, 'i -> i ()') - rearrange(seq, 'j -> () j')
        dist = dist.clamp(-max_pos_emb, max_pos_emb) + max_pos_emb
        rel_pos_emb = self.rel_pos_emb(dist).to(q_r)
        pos_attn_r = einsum('b h n d, n r d -> b h n r', q_r, rel_pos_emb) * self.scale
        pos_attn_i = einsum('b h n d, n r d -> b h n r', q_i, rel_pos_emb) * self.scale

        dots_r = dots_r + pos_attn_r
        dots_i = dots_i + pos_attn_i

        if exists(mask) or exists(context_mask):
            mask = default(mask, lambda: torch.ones(*x.shape[:2], device = device))
            context_mask = default(context_mask, mask) if not has_context else default(context_mask, lambda: torch.ones(*context.shape[:2], device = device))
            mask_value = -torch.finfo(dots.dtype).max
            mask = rearrange(mask, 'b i -> b () i ()') * rearrange(context_mask, 'b j -> b () () j')
            dots_r.masked_fill_(~mask, mask_value)
            dots_i.masked_fill_(~mask, mask_value)
    
        attn = Csoftmax(torch.stack((dots_r, dots_i), dim=-1), dim = -1)
        v_r, v_i = v[...,0], v[...,1]
        
        out_r = einsum('b h i j, b h j d -> b h i d', attn, v_r) 
        out_i = einsum('b h i j, b h j d -> b h i d', attn, v_i) 
        out_r = rearrange(out_r, 'b h n d -> b n (h d)')
        out_i = rearrange(out_i, 'b h n d -> b n (h d)')
        out = self.to_out(torch.stack((out_r, out_i), dim=-1))
        return self.dropout(out)


class CFeedForward(nn.Module):
    def __init__(
        self,
        dim,
        mult = 4,
        dropout = 0.
    ):
        super().__init__()
        self.net = nn.Sequential(
            CLinear(dim, dim * mult),
            CSwish(),
            CDropout(dropout),
            CLinear(dim * mult, dim),
            CDropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class CConformerConvModule(nn.Module):
    def __init__(
        self,
        dim,
        causal = False,
        expansion_factor = 2,
        kernel_size = 31,
        dropout = 0.):
        super().__init__()

        inner_dim = dim * expansion_factor
        padding = calc_same_padding(kernel_size) if not causal else (kernel_size - 1, 0)

        self.net = nn.Sequential(
            CLayerNorm(dim),
            Rearrange('b n c d -> b c n d'),
            CConv1d(dim, inner_dim * 2, 1),
            CGLU(dim=1),
            CDepthWiseConv1d(inner_dim, inner_dim, kernel_size = kernel_size, padding = padding),
            CBatchNorm1d(inner_dim) if not causal else nn.Identity(),
            CSwish(),
            CConv1d(inner_dim, dim, 1),
            Rearrange('b c n d -> b n c d'),
            CDropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

# Conformer Block


class CConformerBlock(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        conv_expansion_factor = 2,
        conv_kernel_size = 31,
        attn_dropout = 0.,
        ff_dropout = 0.,
        conv_dropout = 0.
    ):
        super().__init__()
        self.ff1 = CFeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout)
        self.attn = CAttention(dim = dim, dim_head = dim_head, heads = heads, dropout = attn_dropout)
        self.conv = CConformerConvModule(dim = dim, causal = False, expansion_factor = conv_expansion_factor, kernel_size = conv_kernel_size, dropout = conv_dropout)
        self.ff2 = CFeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout)

        self.attn = CPreNorm(dim, self.attn)
        self.ff1 = Scale(0.5, CPreNorm(dim, self.ff1))
        self.ff2 = Scale(0.5, CPreNorm(dim, self.ff2))

        self.post_norm = CLayerNorm(dim)

    def forward(self, x, mask = None):
        x = self.ff1(x) + x
        x = self.attn(x, mask = mask) + x
        x = self.conv(x) + x
        x = self.ff2(x) + x
        x = self.post_norm(x)
        return x
