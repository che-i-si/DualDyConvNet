import torch
from torch import nn
import torch.nn.functional as F
from utils.util import cal_cnn_outlen

#%% Encoder

class Encoder(nn.Module):
    def __init__(self, kernel_size: int, n_layers: int = 3):
        super(Encoder, self).__init__()
        self.kernel_size = kernel_size
        self.n_layers = n_layers

        self.blocks = nn.ModuleList()
        for i in range(n_layers):
            self.blocks.append(
                self.get_block()
            )

    def forward(self, x):
        """
        x   (B, 1 or F, H, W)
        """
        H, W = x.shape[-2], x.shape[-1]
        for block in self.blocks:
            x = block(x)
        # ===== Interpolate
        x = F.interpolate(x, (H, W), mode='bilinear', align_corners=False)

        return x

    def get_block(self):
        return nn.Sequential(
            nn.AvgPool2d(self.kernel_size, stride=1, padding=0),
            nn.Tanh(),
            nn.MaxPool2d(self.kernel_size, stride=1, padding=0)
        )


#%% MultiScaleBlock

class MultiScaleBlock(nn.Module):
    def __init__(self, F):
        super(MultiScaleBlock, self).__init__()
        # ===== Multi Scale Convolution
        self.conv_1 = nn.Sequential(
            ##### RESHAPE
            nn.Flatten(0, 1),                  # (B, F, H, W) -> (B*F, H, W)
            nn.Unflatten(0, (-1, 1)),        # -> (B*F, 1, H, W)
            ##### Convolution
            nn.Conv2d(1, 1,
                      kernel_size=2, padding=1, bias=False),    # -> (B*F, 1, H', W')
            nn.BatchNorm2d(1),
            nn.Tanh(),
        )
        self.conv_2 = nn.Sequential(
            ##### RESHAPE
            nn.Flatten(0, 1),                  # (B, F, H, W) -> (B*F, H, W)
            nn.Unflatten(0, (-1, 1)),        # -> (B*F, 1, H, W)
            ##### Convolution
            nn.Conv2d(1, 1,
                      kernel_size=4, padding=1, bias=False),    # -> (B*F, 1, H'', W'')
            nn.BatchNorm2d(1),
            nn.Tanh(),
        )
        # ===== Concat Convolution
        self.dot_conv = nn.Sequential(
            nn.Conv2d(2, 1,
                      kernel_size=1, padding=0, bias=False),    # (B*F, 2, H, W) -> (B*F, 1, H, W)
            nn.BatchNorm2d(1),
            ##### RESHAPE
            nn.Flatten(0, 1),                  # -> (B*F, H, W)
            nn.Unflatten(0, (-1, F)),        # -> (B, F, H, W)
        )

    def forward(self, x):
        """
        x    (B, F, H, W)
        """
        H, W = x.shape[-2], x.shape[-1]

        # ===== Multi Scale Feature
        x1 = self.conv_1(x)    # -> (B*F, 1, H', W')
        x2 = self.conv_2(x)    # -> (B*F, 1, H'', W'')

        # ===== Interpolation
        x1 = F.interpolate(x1, (H, W), mode='bilinear', align_corners=False)    # (B*F, 1, H' , W' ) -> (B*F, 1, H, W)
        x2 = F.interpolate(x2, (H, W), mode='bilinear', align_corners=False)    # (B*F, 1, H'', W'') -> (B*F, 1, H, W)

        # ===== Concat & Dot-Convolution
        x = torch.cat((x1, x2), dim=1)      # -> (B*F, 2, H, W)
        del x1, x2
        x = self.dot_conv(x)                        # -> (B, F, H, W)
        return x


#%% Filter Generation

class ChanFilter(nn.Module):
    def __init__(self, F, H, W, chan_k: int, k: int):
        super(ChanFilter, self).__init__()

        # self.conv = nn.Sequential(
        self.chan_wise_conv = nn.Sequential(
            ##### RESAHPE
            nn.Flatten(0, 1),                  # (B, F, H, W) -> (B*F, H, W)
            nn.Unflatten(0, (-1, 1)),        # -> (B*F, 1, H, W)
            ##### DOT CONVOLUTION
            nn.Conv2d(1, chan_k * chan_k,
                      kernel_size=1, padding=0, bias=True),     # -> (B*F, chan_k^2, H, W)
            ##### RESHAPE
            nn.Unflatten(0, (-1, F)),        # -> (B, F, chan_k^2, H, W)
            nn.Flatten(1, 2),                  # -> (B, F*chan_k^2, H, W)
        )

        self.ps = nn.Sequential(
            nn.PixelShuffle(chan_k),                            # (B, F*chan_k^2, H, W) -> (B, F, H*chan_k, W*chan_k)
            nn.AdaptiveAvgPool2d((H, W)),                       # -> (B, F, H, W)
        )

        self.aap_3d = nn.AdaptiveAvgPool3d((1, H, W))

        # self.aap = nn.Sequential(
        #     nn.Conv2d(F, F,
        #               kernel_size=1, padding=0, bias=True),     # (B, F, H, W) -> (B, F, H, W)
        #     nn.AdaptiveAvgPool2d((k, k)),                       # -> (B, F, k, k)
        # )

        self.dot_conv = nn.Conv2d(F, F,
                                    kernel_size=1, padding=0, bias=True)   # (B, F, H, W) -> (B, F, H, W)
        self.aap_2d = nn.AdaptiveAvgPool2d((k, k))                         # -> (B, F, k, k)



    def forward(self, x):
        """
        x    (B, F, H, W)
        """
        x = self.ps(self.chan_wise_conv(x)) + self.aap_3d(x)
        x = self.dot_conv(x)
        x = self.aap_2d(x)
        return x


class SpatFilter(nn.Module):
    def __init__(self, k: int):
        super(SpatFilter, self).__init__()

        self.dot_conv = nn.Conv2d(1, k * k, kernel_size=1, padding=0, bias=True)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        """
        x    (B, 1, H, W)
        """
        x = self.dot_conv(x) + x
        x = self.softmax(x)
        return x

#%% Decoder

class Decoder(nn.Module):
    def __init__(self, H, W, bias=True):
        super(Decoder, self).__init__()
        # ===== CONVOLUTION BLOCK
        self.conv = nn.Sequential(                          # (B, 2, H, W) -> (B, 16, h, w)
            nn.Conv2d(2, 6,
                      5, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, padding=0),

            nn.Conv2d(6, 16,
                      5, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, padding=0),
        )

        # ===== Calculate Output Shape
        h = cal_cnn_outlen(self.conv, H, pos=0)
        w = cal_cnn_outlen(self.conv, W, pos=1)

        # ===== LINEAR BLOCK
        self.linear = nn.Sequential(
            ##### RESHAPE
            nn.Flatten(1, -1),                                # (B, 16, h, w) -> (B, 16 * h * w)
            ##### LINEAR
            nn.Linear(16 * h * w, 16 * h * w, bias=bias),
            nn.ReLU(inplace=True),

            nn.Linear(16 * h * w, (16 * h * w) // 2, bias=bias),
            nn.ReLU(inplace=True),

            nn.Linear((16 * h * w) // 2, 1, bias=bias),     # -> (B, 1)
            nn.Sigmoid()
        )

    def forward(self, x):
        """x    (B, 2, H, W)"""
        x = self.conv(x)
        x = self.linear(x)
        return x

#%% MODEL

class DyFilterTopoNet(nn.Module):
    def __init__(self, args):
        super(DyFilterTopoNet, self).__init__()
        self.args = args
        self.k = args.k
        # -----
        self.encoder = Encoder(kernel_size=args.enc_kernel, n_layers=args.enc_layers)

        self.MSB_spat = MultiScaleBlock(1)
        self.MSB_chan = MultiScaleBlock(args.F)

        self.spatFilter = SpatFilter(args.k)
        self.chanFilter = ChanFilter(args.F, args.H, args.W, args.chan_k, args.k)

        self.dec_pre = Decoder(args.H, args.W)
        self.dec_rec = Decoder(args.H, args.W)

        # -----
        self.unfold = nn.Sequential(
            nn.Unfold((args.k, args.k), padding=(args.k - 1) // 2, stride=1, dilation=1),
            nn.Unflatten(1, (args.F, args.k, args.k))
        )

    def forward(self, x):
        """
        x    (B, F, H, W)
        """
        H, W = x.shape[-2], x.shape[-1]
        # Comprise Frequency Bands
        x1 = F.adaptive_max_pool3d(x, (1, H, W))

        # --------------------------------------------------
        # ===== Re-Mapping Topo
        x = self.encoder(x)         # (B, F, H, W)
        x1 = self.encoder(x1)       # (B, 1, H, W)

        # ===== MultiScale Feature
        chan = self.MSB_chan(x)
        spat = self.MSB_spat(x1)

        # ===== Filter Generation
        chan = self.chanFilter(chan)
        spat = self.spatFilter(spat)

        # ===== Dynamic Convolution
        chan = torch.einsum('bdhwl, bdhw -> bl', self.unfold(x), chan).unsqueeze(1)                # Chan Feature
        spat = torch.einsum('bdhwl, bohwl -> bol', self.unfold(x), self.prepare_spatFilter(spat, H, W))  # Spat Feature

        x = torch.cat((spat, chan), dim=1).reshape(-1, 2, H, W)
        del x1, spat, chan

        # ===== Regression
        pre = self.dec_pre(x)
        self.copy_bias()  # bias 복사

        rec = self.dec_rec(x)
        del x

        return pre, rec

    ################################################
    def copy_bias(self):
        for m1, m2 in zip(self.dec_pre.modules(), self.dec_rec.modules()):
            if isinstance(m1, nn.Conv2d) or isinstance(m1, nn.Linear):
                m2.bias.data = m1.bias.data     # copy bias
                m2.bias.requires_grad = False   # 학습하지 않도록 고정

    ################################################
    def prepare_spatFilter(self, filter, H, W):
        return filter.reshape(-1, self.k, self.k, H*W).unsqueeze(1)
