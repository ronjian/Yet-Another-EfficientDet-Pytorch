# Author: Zylo117

import math

from torch import nn
import torch.nn.functional as F
import torch


class Conv2dStaticSamePadding(nn.Module):
    """
    created by Zylo117
    The real keras/tensorflow conv2d with same padding
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, groups=1, dilation=1, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                              bias=bias, groups=groups)
        self.stride = self.conv.stride
        self.kernel_size = self.conv.kernel_size
        self.dilation = self.conv.dilation

        if isinstance(self.stride, int):
            self.stride = [self.stride] * 2
        elif len(self.stride) == 1:
            self.stride = [self.stride[0]] * 2

        if isinstance(self.kernel_size, int):
            self.kernel_size = [self.kernel_size] * 2
        elif len(self.kernel_size) == 1:
            self.kernel_size = [self.kernel_size[0]] * 2

    def forward(self, x):
        h, w = x.shape[-2:]

        # extra_h = (math.ceil(w / self.stride[1]) - 1) * self.stride[1] - w + self.kernel_size[1]
        # extra_v = (math.ceil(h / self.stride[0]) - 1) * self.stride[0] - h + self.kernel_size[0]
        
        old_extra_h = (math.ceil(w / self.stride[1]) - 1) * self.stride[1] - w + self.kernel_size[1]
        old_extra_v = (math.ceil(h / self.stride[0]) - 1) * self.stride[0] - h + self.kernel_size[0]
        if self.kernel_size[0] == 3:
            if self.stride[0] == 2:
                extra_h, extra_v = 1, 1
            else:
                extra_h, extra_v = 2, 2
        elif self.kernel_size[0] == 1:
            extra_h, extra_v = 0, 0
        elif self.kernel_size[0] == 5:
            if self.stride[0] == 2:
                extra_h, extra_v = 3, 3
            else:
                extra_h, extra_v = 4, 4
        if extra_h != old_extra_h or extra_v != old_extra_v:
            print(w, h, self.stride, self.kernel_size, extra_h, extra_v, old_extra_h, old_extra_v)
            exit()
        
        left = extra_h // 2
        right = extra_h - left
        top = extra_v // 2
        bottom = extra_v - top

        # x = F.pad(x, [left, right, top, bottom])
        x = torch.constant_pad_nd(x,(left, right, top, bottom))

        x = self.conv(x)
        return x


class MaxPool2dStaticSamePadding(nn.Module):
    """
    created by Zylo117
    The real keras/tensorflow MaxPool2d with same padding
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.pool = nn.MaxPool2d(*args, **kwargs)
        self.stride = self.pool.stride
        self.kernel_size = self.pool.kernel_size

        if isinstance(self.stride, int):
            self.stride = [self.stride] * 2
        elif len(self.stride) == 1:
            self.stride = [self.stride[0]] * 2

        if isinstance(self.kernel_size, int):
            self.kernel_size = [self.kernel_size] * 2
        elif len(self.kernel_size) == 1:
            self.kernel_size = [self.kernel_size[0]] * 2

    def forward(self, x):
        h, w = x.shape[-2:]
        
        # extra_h = (math.ceil(w / self.stride[1]) - 1) * self.stride[1] - w + self.kernel_size[1]
        # extra_v = (math.ceil(h / self.stride[0]) - 1) * self.stride[0] - h + self.kernel_size[0]

        old_extra_h = (math.ceil(w / self.stride[1]) - 1) * self.stride[1] - w + self.kernel_size[1]
        old_extra_v = (math.ceil(h / self.stride[0]) - 1) * self.stride[0] - h + self.kernel_size[0]
        if self.kernel_size[0] == 3:
            if self.stride[0] == 2:
                extra_h, extra_v = 1, 1
            else:
                extra_h, extra_v = 2, 2
        elif self.kernel_size[0] == 1:
            extra_h, extra_v = 0, 0
        elif self.kernel_size[0] == 5:
            if self.stride[0] == 2:
                extra_h, extra_v = 3, 3
            else:
                extra_h, extra_v = 4, 4
        if extra_h != old_extra_h or extra_v != old_extra_v:
            print(w, h, self.stride, self.kernel_size, extra_h, extra_v, old_extra_h, old_extra_v)
            exit()

        left = extra_h // 2
        right = extra_h - left
        top = extra_v // 2
        bottom = extra_v - top

        # x = F.pad(x, [left, right, top, bottom])
        x = torch.constant_pad_nd(x,(left, right, top, bottom))

        x = self.pool(x)
        return x
