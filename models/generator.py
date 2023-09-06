from models.conformer import CConformerBlock
from utils import *


class UniDeepFsmn(nn.Module):

    def __init__(self, input_dim, output_dim, lorder=None, hidden_size=None):
        super(UniDeepFsmn, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        if lorder is None:
            return

        self.lorder = lorder
        self.hidden_size = hidden_size

        self.linear = nn.Linear(input_dim, hidden_size)

        self.project = nn.Linear(hidden_size, output_dim, bias=False)

        self.conv1 = nn.Conv2d(output_dim, output_dim, [lorder, 1], [1, 1], groups=output_dim, bias=False)

    def forward(self, input):
        ## input: batch (b) x sequence(T) x feature (h)
        f1 = F.relu(self.linear(input))

        p1 = self.project(f1)

        x = torch.unsqueeze(p1, 1)
        #x: batch (b) x channel (c) x sequence(T) x feature (h)
        x_per = x.permute(0, 3, 2, 1)
        #x_per: batch (b) x feature (h) x sequence(T) x channel (c)
        y = F.pad(x_per, [0, 0, self.lorder - 1, 0])

        out = x_per + self.conv1(y)

        out1 = out.permute(0, 3, 2, 1)
        #out1: batch (b) x channel (c) x sequence(T) x feature (h)
        return input + out1.squeeze(1)

class CFsmn(nn.Module):

    def __init__(self, nIn, nHidden=128, nOut=128):
        super(CFsmn, self).__init__()

        self.fsmn_re_L1 = UniDeepFsmn(nIn, nHidden, 20, nHidden)
        self.fsmn_im_L1 = UniDeepFsmn(nIn, nHidden, 20, nHidden)

    def forward(self, x):
        # # shpae of input x : [b,c,h,T,2], [6, 256, 1, 106, 2]
        b,c,T,h,d = x.size()
        #x : [b,T,h,c,2]
        x = x.permute(0, 2, 3, 1, 4)
        x = torch.reshape(x, (b*T, h, c, d))

        real = self.fsmn_re_L1(x[..., 0]) - self.fsmn_im_L1(x[..., 1])
        imaginary = self.fsmn_re_L1(x[..., 1]) + self.fsmn_im_L1(x[..., 0])
        # output: [b*T,h,c,2], [6*106, h, 256, 2]
        output = torch.stack((real, imaginary), dim=-1)

        output = torch.reshape(output, (b, T, h, c, d))
        # output: [b,c,h,T,2], [6, 99, 1024, 2]
        #output = torch.transpose(output, 1, 3)

        return output.permute(0, 3, 1, 2, 4)

class CConstantPad2d(nn.Module):
    def __init__(self, padding, value):
        super(CConstantPad2d, self).__init__()
        self.padding = padding
        self.value = value
        self.pad_r = nn.ConstantPad2d(self.padding, self.value)
        self.pad_i = nn.ConstantPad2d(self.padding, self.value)

    def forward(self, x):
        a, b = x[...,0], x[...,1]
        return torch.stack((self.pad_r(a), self.pad_i(b)), dim=-1)

class CConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, use_fsmn=True, **kwargs):
        super().__init__()
        self.use_fsmn = use_fsmn
        if use_fsmn:
            self.fsmn = CFsmn(nIn=out_channel, nHidden=out_channel, nOut=out_channel)
        ## Model components
        self.conv_r = nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride, padding=padding,
                                 dilation=dilation, groups=groups, bias=bias, **kwargs)
        self.conv_i = nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride, padding=padding,
                                 dilation=dilation, groups=groups, bias=bias, **kwargs)

    def forward(self, x):
        a, b = x[...,0], x[...,1]
        real = self.conv_r(a) - self.conv_i(b)
        imag = self.conv_r(b) + self.conv_i(a)
        out = torch.stack((real, imag), dim=-1)
        if self.use_fsmn:
            out = self.fsmn(out)
        return out

class CInstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=False):
        super().__init__()
        self.in_r = nn.InstanceNorm2d(num_features=num_features, eps=eps, momentum=momentum, affine=affine)
        self.in_i = nn.InstanceNorm2d(num_features=num_features, eps=eps, momentum=momentum, affine=affine)

    def forward(self, x):
        a = self.in_r(x[...,0])
        b = self.in_i(x[...,1])
        return torch.stack((a, b), dim=-1)

class CDilatedDenseNet(nn.Module):
    def __init__(self, depth=4, in_channels=64):
        super(CDilatedDenseNet, self).__init__()
        self.depth = depth
        self.in_channels = in_channels
        self.twidth = 2
        self.kernel_size = (self.twidth, 3)
        for i in range(self.depth):
            dil = 2 ** i
            pad_length = self.twidth + (dil - 1) * (self.twidth - 1) - 1
            setattr(self, 'pad{}'.format(i + 1), CConstantPad2d((1, 1, pad_length, 0), value=0.))
            setattr(self, 'conv{}'.format(i + 1),
                    CConv2d(self.in_channels * (i + 1), self.in_channels, kernel_size=self.kernel_size,
                              dilation=(dil, 1), use_fsmn=False))
            setattr(self, 'norm{}'.format(i + 1), CInstanceNorm2d(in_channels, affine=True))
            setattr(self, 'prelu{}'.format(i + 1), nn.PReLU(self.in_channels))
            setattr(self, 'cfsmn{}'.format(i + 1), CFsmn(nIn=self.in_channels, nHidden=self.in_channels, nOut=self.in_channels))

    def forward(self, x):
        skip = x
        for i in range(self.depth):
            out = getattr(self, 'pad{}'.format(i + 1))(skip)
            out = getattr(self, 'conv{}'.format(i + 1))(out)
            out = getattr(self, 'norm{}'.format(i + 1))(out)
            out = getattr(self, 'prelu{}'.format(i + 1))(out)            
            out = getattr(self, 'cfsmn{}'.format(i + 1))(out)
            skip = torch.cat([out, skip], dim=1)            
        return out


class CDenseEncoder(nn.Module):
    def __init__(self, in_channel, channels=64):
        super(CDenseEncoder, self).__init__()
        self.conv_1 = nn.Sequential(
            CConv2d(in_channel, channels, (1, 1), (1, 1), use_fsmn=False),
            CInstanceNorm2d(channels, affine=True),
            nn.PReLU(channels)
        )
        self.dilated_dense = CDilatedDenseNet(depth=4, in_channels=channels)
        self.conv_2 = nn.Sequential(
            CConv2d(channels, channels, (1, 3), (1, 2), padding=(0, 1), use_fsmn=False),
            CInstanceNorm2d(channels, affine=True),
            nn.PReLU(channels)
        )

    def forward(self, x):
        x = self.conv_1(x)
        x = self.dilated_dense(x)
        x = self.conv_2(x)
        return x


class TSCB(nn.Module):
    def __init__(self, num_channel=64):
        super(TSCB, self).__init__()
        self.time_conformer = CConformerBlock(dim=num_channel, dim_head=num_channel//4, heads=4,
                                             conv_kernel_size=31, attn_dropout=0.2, ff_dropout=0.2)
        self.freq_conformer = CConformerBlock(dim=num_channel, dim_head=num_channel//4, heads=4,
                                             conv_kernel_size=31, attn_dropout=0.2, ff_dropout=0.2)

    def forward(self, x_in):
        #x_in: torch.Size([1, 64, 321, 101, 2])
        b, c, t, f, d = x_in.size()
        x_t = x_in.permute(0, 3, 2, 1, 4).contiguous().view(b*f, t, c, d)
        x_t = self.time_conformer(x_t) + x_t
        x_f = x_t.view(b, f, t, c, d).permute(0, 2, 1, 3, 4).contiguous().view(b*t, f, c, d)
        x_f = self.freq_conformer(x_f) + x_f
        x_f = x_f.view(b, t, f, c, d).permute(0, 3, 1, 2, 4)
        return x_f


class CSPConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, r=1):
        super(CSPConvTranspose2d, self).__init__()
        self.pad1 = CConstantPad2d((1, 1, 0, 0), value=0.)
        self.out_channels = out_channels
        self.conv = CConv2d(in_channels, out_channels * r, kernel_size=kernel_size, stride=(1, 1), use_fsmn=False)
        self.r = r

    def forward(self, x):
        x = self.pad1(x)
        out = self.conv(x)
        batch_size, nchannels, H, W, C = out.shape
        out = out.view((batch_size, self.r, nchannels // self.r, H, W, C))
        out = out.permute(0, 2, 3, 4, 1, 5)
        out = out.contiguous().view((batch_size, nchannels // self.r, H, -1, C))
        return out

class CMaskDecoder(nn.Module):
    def __init__(self, num_features, num_channel=64, out_channel=1):
        super(CMaskDecoder, self).__init__()
        self.dense_block = CDilatedDenseNet(depth=4, in_channels=num_channel)
        self.sub_pixel = CSPConvTranspose2d(num_channel, num_channel, (1, 3), 2)
        self.conv_1 = CConv2d(num_channel, out_channel, (1, 2), use_fsmn=False)
        self.norm = CInstanceNorm2d(out_channel, affine=True)
        self.relu = nn.LeakyReLU(inplace=True)
        self.final_conv = CConv2d(out_channel, out_channel, (1, 1), use_fsmn=False)

    def forward(self, x):
        x = self.dense_block(x)
        x = self.sub_pixel(x)
        x = self.conv_1(x)
        x = self.relu(self.norm(x))
        x = self.final_conv(x)
        return torch.tanh(x)

class ComplexDecoder(nn.Module):
    def __init__(self, num_channel=64):
        super(ComplexDecoder, self).__init__()
        self.dense_block = CDilatedDenseNet(depth=4, in_channels=num_channel)
        self.sub_pixel = CSPConvTranspose2d(num_channel, num_channel, (1, 3), 2)
        self.prelu = nn.PReLU(num_channel)
        self.norm = CInstanceNorm2d(num_channel, affine=True)
        self.conv = CConv2d(num_channel, 1, (1, 2), use_fsmn=False)

    def forward(self, x):
        x = self.dense_block(x)
        x = self.sub_pixel(x)
        x = self.prelu(self.norm(x))
        x = self.conv(x)
        return x


class TSCNet(nn.Module):
    def __init__(self, num_channel=64, num_features=201):
        super(TSCNet, self).__init__()
        self.alpha = 0.75
        self.beta = 0.25
        self.dense_encoder = CDenseEncoder(in_channel=1, channels=num_channel)

        self.TSCB_1 = TSCB(num_channel=num_channel)
        self.TSCB_2 = TSCB(num_channel=num_channel)
        self.TSCB_3 = TSCB(num_channel=num_channel)

        self.mask_decoder = CMaskDecoder(num_features, num_channel=num_channel, out_channel=1)
        self.complex_decoder = ComplexDecoder(num_channel=num_channel)

    def forward(self, x):
        noisy_phase = torch.angle(torch.complex(x[:, 0, :, :], x[:, 1, :, :])).unsqueeze(1)
        x_in = x.permute(0,2,3,1).unsqueeze(1)
        a, b = x_in[...,0], x_in[...,1]

        out_1 = self.dense_encoder(x_in)
        out_2 = self.TSCB_1(out_1)
        out_3 = self.TSCB_2(out_2)
        out_4 = self.TSCB_3(out_3)

        cmask = self.mask_decoder(out_4)
        c, d = cmask[...,0], cmask[...,1]

        masked_r = a*c - b*d
        masked_i = a*d + b*c
        complex_out = self.complex_decoder(out_4)
        
        final_real = self.alpha * masked_r + self.beta * complex_out[..., 0]
        final_imag = self.alpha * masked_i + self.beta * complex_out[..., 1]

        return final_real, final_imag, torch.squeeze(cmask, 1)
