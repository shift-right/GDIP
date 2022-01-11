import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch.nn import init

class ConvGuidedFilter(nn.Module): # Deep Guided Filter
    def __init__(self, radius=1, channels =3 ,norm=nn.BatchNorm2d):
        super(ConvGuidedFilter, self).__init__()
        self.channels = channels
        self.box_filter = nn.Conv2d(channels, channels, kernel_size=3, padding=radius, dilation=radius, bias=False, groups=channels)
        self.conv_a = nn.Sequential(nn.Conv2d(2*channels, 32, kernel_size=1, bias=False),
                                    norm(32),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(32, 32, kernel_size=1, bias=False),
                                    norm(32),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(32, channels, kernel_size=1, bias=False))
        self.box_filter.weight.data[...] = 1.0

    def forward(self, x_lr, y_lr, x_hr):

        _,_, h_lrx, w_lrx = x_lr.size()
        _,_, h_hrx, w_hrx = x_hr.size()

        N = self.box_filter(x_lr.data.new().resize_((1, self.channels, h_lrx, w_lrx)).fill_(1.0))
        ## mean_x
        mean_x = self.box_filter(x_lr)/N
        ## mean_y
        mean_y = self.box_filter(y_lr)/N
        ## cov_xy
        cov_xy = self.box_filter(x_lr * y_lr)/N - mean_x * mean_y
        ## var_x
        var_x  = self.box_filter(x_lr * x_lr)/N - mean_x * mean_x

        ## A
        A = self.conv_a(torch.cat([cov_xy, var_x], dim=1))
        ## b
        b = mean_y - A * mean_x

        ## mean_A; mean_b
        mean_A = F.interpolate(A, (h_hrx, w_hrx), mode='bilinear', align_corners=True)
        mean_b = F.interpolate(b, (h_hrx, w_hrx), mode='bilinear', align_corners=True)

        return mean_A * x_hr + mean_b

class AdaptiveNorm(nn.Module):
    def __init__(self, n):
        super(AdaptiveNorm, self).__init__()

        self.w_0 = nn.Parameter(torch.Tensor([1.0]))
        self.w_1 = nn.Parameter(torch.Tensor([0.0]))

        self.bn  = nn.BatchNorm2d(n, momentum=0.999, eps=0.001)

    def forward(self, x):
        return self.w_0 * x + self.w_1 * self.bn(x)


class DeepGuidedFilterConvGF(nn.Module):
    def __init__(self, radius=1, channels =3):
        super(DeepGuidedFilterConvGF, self).__init__()
        self.gf = ConvGuidedFilter(radius, channels = channels, norm=AdaptiveNorm)

    def forward(self, x_lr, x_hr):
        return self.gf(x_lr, self.lr(x_lr), x_hr).clamp(0, 1)

class DeepGuidedFilterGuidedMapConvGF(nn.Module):
    def __init__(self, radius=1, channels =3, dilation=0, c=16):
        super(DeepGuidedFilterGuidedMapConvGF, self).__init__()

        self.gf = ConvGuidedFilter(radius=radius, channels = channels, norm=AdaptiveNorm)
        self.guided_map = nn.Sequential(
            nn.Conv2d(channels, c, 1, bias=False) if dilation==0 else \
                nn.Conv2d(channels, c, 3, padding=dilation, dilation=dilation, bias=False),
            AdaptiveNorm(c),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c, channels, 1)
        )

    def forward(self,x_lr, y_lr, x_hr):
        return self.gf(self.guided_map(x_lr), y_lr, self.guided_map(x_hr)).clamp(0, 1)