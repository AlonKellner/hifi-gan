import torch
import torch.nn as nn
from torch.nn import Conv1d, ConvTranspose1d, LeakyReLU, Tanh, Conv2d, ConvTranspose2d
from torch.nn.utils import weight_norm, spectral_norm

from src.utils import init_weights
from extra_utils import get_padding, get_padding_trans


def get_model_from_config(model_config):
    return nn.Sequential(*get_modules_from_config(model_config))


def get_modules_from_config(modules_configs):
    return [get_module_from_config(block_config) for block_config in modules_configs]


def get_module_from_config(block_config):
    if len(block_config) == 1:
        block_name = block_config[0]
        if block_name == "tanh":
            return Tanh()
    if len(block_config) == 2:
        block_name, block_parameters = block_config
        if block_name == "conv":
            chin, chout, kernel, stride, norm = process_conv_params(*block_parameters)
            layer = norm(
                Conv1d(chin, chout, kernel, stride, padding=get_padding(kernel, stride=stride, dilation=1))
            )
            layer.apply(init_weights)
            return layer
        elif block_name == "conv2":
            chin, chout, kernel, stride, norm = process_conv_params(*block_parameters)
            layer = norm(
                Conv2d(chin, chout, kernel, stride, padding=get_padding(kernel, stride=stride, dilation=(1, 1)))
            )
            layer.apply(init_weights)
            return layer
        elif block_name == "trans":
            chin, chout, kernel, stride, norm = process_conv_params(*block_parameters)
            padding, output_padding = get_padding_trans(kernel, stride=stride, dilation=1)
            layer = norm(
                ConvTranspose1d(chin, chout, kernel, stride, padding=padding, output_padding=output_padding)
            )
            layer.apply(init_weights)
            return layer
        elif block_name == "trans2":
            chin, chout, kernel, stride, norm = process_conv_params(*block_parameters)
            padding, output_padding = get_padding_trans(kernel, stride=stride, dilation=(1, 1))
            layer = norm(
                ConvTranspose2d(chin, chout, kernel, stride, padding=padding, output_padding=output_padding)
            )
            layer.apply(init_weights)
            return layer
        elif block_name == "res_fusion":
            chin, blocks = block_parameters
            return Conv1dResBlockFusion(chin, blocks)
        elif block_name == "lrelu":
            slope = block_parameters
            return LeakyReLU(slope)


def process_conv_params(chin, chout, kernel, stride, norm_type=None):
    norm = weight_norm
    if norm_type == 'spectral':
        norm = spectral_norm
    return chin, chout, kernel, stride, norm


class ConfigurableModel(torch.nn.Module):
    def __init__(self, model_config):
        super(ConfigurableModel, self).__init__()

        self.model = get_model_from_config(model_config)

    def forward(self, x):
        return self.model(x)


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
        self.model = ConfigurableModel([("conv", (channels, channels, *layer)) for layer in layers])

    def forward(self, x):
        delta = self.model(x)
        x = x + delta
        return x
