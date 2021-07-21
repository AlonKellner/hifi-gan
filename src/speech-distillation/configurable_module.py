import torch
import torch.nn as nn
from torch.nn import Conv1d, ConvTranspose1d, LeakyReLU, Tanh, Conv2d, ConvTranspose2d, AvgPool1d
from torch.nn.utils import weight_norm, spectral_norm

from src.utils import init_weights

from extra_utils import get_padding, get_padding_trans
from custom_layers import Conv1dRechanneled, Period1d
from custom_blocks import ResBlock, FusionBlock


def get_module_from_config(module_config):
    if isinstance(module_config, list):
        module_config = module_config
        module = nn.Sequential(*map(get_module_from_config, module_config))
        return module
    if len(module_config) == 1:
        return get_no_params_module_from_config(*module_config)
    if len(module_config) >= 2:
        return get_with_params_module_from_config(*module_config)


def get_no_params_module_from_config(module_name, should_extract_features=False):
    if module_name == 'tanh':
        return Tanh()


def get_with_params_module_from_config(module_name, module_parameters, should_extract_features=False):
    if module_name == 'conv':
        chin, chout, kernel, stride, dilation, groups, norm = process_conv_params(*module_parameters)
        layer = norm(
            Conv1d(chin, chout,
                   kernel_size=kernel,
                   stride=stride,
                   dilation=dilation,
                   groups=groups,
                   padding=get_padding(kernel, stride=stride, dilation=dilation)
                   )
        )
        layer.apply(init_weights)
        return layer
    if module_name == 'conv_rech':
        chin, chout, kernel, stride, dilation, groups, norm = process_conv_params(*module_parameters)
        layer = norm(
            Conv1dRechanneled(chin, chout,
                              kernel_size=kernel,
                              stride=stride,
                              dilation=dilation,
                              groups=groups,
                              padding=get_padding(kernel, stride=stride, dilation=dilation)
                              )
        )
        layer.apply(init_weights)
        return layer
    elif module_name == 'conv2':
        chin, chout, kernel, stride, dilation, groups, norm = process_conv_params(*module_parameters)
        layer = norm(
            Conv2d(chin, chout,
                   kernel_size=kernel,
                   stride=stride,
                   dilation=dilation,
                   groups=groups,
                   padding=get_padding(kernel, stride=stride, dilation=dilation)
                   )
        )
        layer.apply(init_weights)
        return layer
    elif module_name == 'trans':
        chin, chout, kernel, stride, dilation, groups, norm = process_conv_params(*module_parameters)
        padding, output_padding = get_padding_trans(kernel, stride=stride, dilation=dilation)
        layer = norm(
            ConvTranspose1d(chin, chout,
                            kernel_size=kernel,
                            stride=stride,
                            dilation=dilation,
                            groups=groups,
                            padding=padding,
                            output_padding=output_padding
                            )
        )
        layer.apply(init_weights)
        return layer
    elif module_name == 'trans2':
        chin, chout, kernel, stride, dilation, groups, norm = process_conv_params(*module_parameters)
        padding, output_padding = get_padding_trans(kernel, stride=stride, dilation=(1, 1))
        layer = norm(
            ConvTranspose2d(chin, chout,
                            kernel_size=kernel,
                            stride=stride,
                            dilation=dilation,
                            groups=groups,
                            padding=padding,
                            output_padding=output_padding
                            )
        )
        layer.apply(init_weights)
        return layer
    elif module_name == 'pool':
        kernel, stride = module_parameters
        layer = AvgPool1d(kernel_size=kernel,
                          stride=stride,
                          padding=get_padding(kernel, stride=stride, dilation=1)
                          )
        return layer
    elif module_name == 'period':
        period, padding_mode, padding_value = process_period_params(*module_parameters)
        layer = Period1d(period=period,
                         padding_mode=padding_mode,
                         padding_value=padding_value)
        return layer
    elif module_name == 'fusion':
        modules_configs = module_parameters
        modules = nn.ModuleList(list(map(ConfigurableModule, modules_configs)))
        return FusionBlock(modules)
    elif module_name == 'res':
        module_config = module_parameters
        module = ConfigurableModule(module_config)
        return ResBlock(module)
    elif module_name == 'lrelu':
        slope = module_parameters
        return LeakyReLU(slope)


def process_conv_params(chin, chout, kernel, stride=1, dilation=1, groups=1, norm_type=None):
    norm = weight_norm
    if norm_type == 'spectral':
        norm = spectral_norm
    return chin, chout, kernel, stride, dilation, groups, norm


def process_period_params(period, padding_mode='constant', padding_value=0):
    return period, padding_mode, padding_value


class ConfigurableModule(torch.nn.Module):
    def __init__(self, module_config):
        super(ConfigurableModule, self).__init__()

        self.module = get_module_from_config(module_config)

    def forward(self, x):
        return self.module(x)
