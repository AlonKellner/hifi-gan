import torch
import torch.nn as nn
from torch.nn import Conv1d, ConvTranspose1d, LeakyReLU, Tanh
from torch.nn.utils import weight_norm

from blocks import Conv1dResBlockFusion
from src.utils import init_weights
from extra_utils import get_padding, get_padding_trans


def get_model_from_config(model_config):
    return nn.Sequential(*[
        get_block_from_config(block_config) for block_config in model_config
    ])


def get_block_from_config(block_config):
    if len(block_config) == 1:
        block_name = block_config[0]
        if block_name == "tanh":
            return Tanh()
    if len(block_config) == 2:
        block_name, block_parameters = block_config
        if block_name == "conv":
            chin, chout, kernel, stride = block_parameters
            layer = weight_norm(Conv1d(chin, chout, kernel, stride, padding=get_padding(kernel, stride=stride, dilation=1)))
            layer.apply(init_weights)
            return layer
        elif block_name == "trans":
            chin, chout, kernel, stride = block_parameters
            padding, output_padding = get_padding_trans(kernel, stride=stride, dilation=1)
            layer = weight_norm(ConvTranspose1d(chin, chout, kernel, stride,
                                                padding=padding, output_padding=output_padding))
            layer.apply(init_weights)
            return layer
        elif block_name == "res_fusion":
            chin, blocks = block_parameters
            return Conv1dResBlockFusion(chin, blocks)
        elif block_name == "lrelu":
            slope = block_parameters
            return LeakyReLU(slope)


class ConfigurableModel(torch.nn.Module):
    def __init__(self, model_config):
        super(ConfigurableModel, self).__init__()

        self.model = get_model_from_config(model_config)

    def forward(self, x):
        return self.model(x)
