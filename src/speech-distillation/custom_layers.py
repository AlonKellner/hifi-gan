import torch
import torch.nn as nn
import torch.nn.functional as f
import math
from extra_utils import get_padding_period, get_padding
from src.meldataset import mel_spectrogram


class Conv1dRechanneled(nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=None, padding=0, dilation=1, groups=1, bias=True,
                 padding_mode='zeros', device=None, dtype=None):
        common_denominator = math.gcd(in_channels, out_channels)
        if stride is None:
            stride = out_channels // common_denominator
        self.rechanneled = out_channels
        super(Conv1dRechanneled, self).__init__(
            in_channels=in_channels,
            out_channels=stride * in_channels,
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
        return self._conv_forward(input, self.weight, self.bias) \
            .transpose(1, 2) \
            .reshape(input.size()[0], -1, self.rechanneled) \
            .transpose(1, 2)


class GroupShuffle1d(nn.Module):
    def __init__(self, groups):
        self.groups = groups
        super(GroupShuffle1d, self).__init__()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        batch_size, channels, length = input.size()
        return input\
            .reshape(batch_size, self.groups, channels//self.groups, -1)\
            .transpose(1, 2)\
            .reshape(batch_size, channels, -1)


class Period1d(nn.Module):
    def __init__(self, period, padding_mode='constant', padding_value=0):
        self.period = period
        self.padding_mode = padding_mode
        self.padding_value = padding_value
        super(Period1d, self).__init__()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        batch_size, channels, length = input.size()
        pre_padding, post_padding = get_padding_period(length, self.period)
        return f.pad(input, (pre_padding, post_padding), self.padding_mode, self.padding_value) \
            .transpose(1, 2) \
            .reshape(batch_size, -1, channels * self.period) \
            .transpose(1, 2)


class AvgPool1dDilated(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1, ceil_mode: bool = False,
                 count_include_pad: bool = True):
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad
        super(AvgPool1dDilated, self).__init__()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        batch_size, channels, length = input.size()
        stacked_input = input\
            .transpose(1, 2)\
            .reshape(batch_size, -1, self.dilation, channels)\
            .transpose(3, 1)
        pooled = f.avg_pool2d(stacked_input, (1, self.kernel_size), (self.stride, 1),
                              (0, self.padding), self.ceil_mode, self.count_include_pad)
        return pooled\
            .transpose(1, 3)\
            .reshape(batch_size, length, channels)\
            .transpose(2, 1)


class MelSpectrogram(nn.Module):
    def __init__(self, sampling_rate, output_channels, kernel_size, stride, padding_mode='constant', padding_value=0):
        self.sampling_rate = sampling_rate
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding_mode = padding_mode
        self.padding_value = padding_value
        super(MelSpectrogram, self).__init__()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        batch_size, channels, length = input.size()
        pre_padding, post_padding = get_padding_period(length, self.stride)
        padded_input = f.pad(input, (pre_padding, post_padding), self.padding_mode, self.padding_value)
        spec = mel_spectrogram(padded_input.squeeze(1),
                               n_fft=self.kernel_size,
                               num_mels=self.output_channels,
                               sampling_rate=self.sampling_rate,
                               hop_size=self.stride,
                               win_size=self.kernel_size,
                               fmin=0,
                               fmax=None
                               )
        return spec


class Noise1d(nn.Module):
    def __init__(self, channels):
        self.channels = channels
        super(Noise1d, self).__init__()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        batch_size, channels, length = input.size()
        return torch.randn(batch_size, self.channels, length)
