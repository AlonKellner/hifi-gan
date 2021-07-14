import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv1d
from torch.nn.utils import weight_norm, remove_weight_norm

from src.utils import init_weights, get_padding

LRELU_SLOPE = 0.1


class Conv1dResBlockFusion(torch.nn.Module):
    def __init__(self, channels, blocks=[[(5, 1), (5, 1)], [(3, 1), (3, 1)]]):
        super(Conv1dResBlockFusion, self).__init__()
        self.res_blocks = nn.ModuleList([
            weight_norm(
                Conv1dResBlock(channels, channels, layers)
            ) for layers in blocks
        ])

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

    def remove_weight_norm(self):
        for block in self.res_blocks:
            block.remove_weight_norm()


class Conv1dResBlock(torch.nn.Module):
    def __init__(self, channels, layers=[(3, 1), (3, 1)]):
        super(Conv1dResBlock, self).__init__()
        self.res_layers = nn.ModuleList([
            weight_norm(
                Conv1d(channels, channels,
                       kernel_size=kernel, stride=1,
                       dilation=dilation,
                       padding=get_padding(kernel, dilation)
                       )
            ) for (kernel, dilation) in layers
        ])
        self.res_layers.apply(init_weights)

    def forward(self, x):
        residual = x
        for res_layer in self.res_layers:
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = res_layer(x)
        x = residual + x
        return x

    def remove_weight_norm(self):
        for layer in self.res_layers:
            remove_weight_norm(layer)
