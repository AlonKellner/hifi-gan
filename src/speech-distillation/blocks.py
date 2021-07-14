import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv1d
from torch.nn.utils import weight_norm

from src.utils import init_weights
from extra_utils import get_padding

LRELU_SLOPE = 0.1


class Conv1dResBlockFusion(torch.nn.Module):
    def __init__(self, channels, blocks=[[[(5, 1), (5, 1)], [(5, 1), (5, 1)]], [[(5, 5), (5, 1)], [(5, 5), (5, 1)]]]):
        super(Conv1dResBlockFusion, self).__init__()
        self.res_blocks = nn.ModuleList([Conv1dResBlock(channels, res_layers) for res_layers in blocks])

    def forward(self, x):
        xsum = None
        for res_block in self.res_blocks:
            value = res_block(x)
            if xsum is None:
                xsum = value
            else:
                xsum += value
        x = xsum / len(self.res_blocks)
        return x


class Conv1dResBlock(torch.nn.Module):
    def __init__(self, channels, res_layers=[[(5, 1), (5, 1)], [(3, 1), (3, 1)]]):
        super(Conv1dResBlock, self).__init__()
        self.block = nn.Sequential(*[Conv1dResLayer(channels, layers) for layers in res_layers])

    def forward(self, x):
        return self.block(x)


class Conv1dResLayer(torch.nn.Module):
    def __init__(self, channels, layers=[(3, 1), (3, 1)]):
        super(Conv1dResLayer, self).__init__()
        self.layers = nn.ModuleList([
            weight_norm(
                Conv1d(channels, channels,
                       kernel_size=kernel, stride=1,
                       dilation=dilation,
                       padding=get_padding(kernel, stride=1, dilation=dilation)
                       )
            ) for (kernel, dilation) in layers
        ])
        self.layers.apply(init_weights)

    def forward(self, x):
        residual = x
        for layer in self.layers:
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = layer(x)
        x = residual + x
        return x
