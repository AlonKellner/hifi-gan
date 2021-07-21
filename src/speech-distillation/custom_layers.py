import torch
import torch.nn as nn
import torch.nn.functional as f
from extra_utils import get_padding_period


class Conv1dRechanneled(nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=None, padding=0, dilation=1, groups=1, bias=True,
                 padding_mode='zeros', device=None, dtype=None):
        if stride is None:
            stride = out_channels
        self.rechanneled = out_channels
        super(Conv1dRechanneled, self).__init__(
            in_channels=in_channels,
            out_channels=stride*in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self._conv_forward(input, self.weight, self.bias)\
            .transpose(1, 2)\
            .reshape(input.size()[0], -1, self.rechanneled)\
            .transpose(1, 2)


class Period1d(nn.Module):
    def __init__(self, period, padding_mode='constant', padding_value=0):
        self.period = period
        self.padding_mode = padding_mode
        self.padding_value = padding_value
        super(Period1d, self).__init__()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        batch_size, channels, length = input.size()
        pre_padding, post_padding = get_padding_period(length, self.period)
        return f.pad(input, (pre_padding, post_padding), self.padding_mode, self.padding_value)\
            .transpose(1, 2)\
            .reshape(batch_size, -1, channels*self.period)\
            .transpose(1, 2)
