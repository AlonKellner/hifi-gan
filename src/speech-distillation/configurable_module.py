import torch
import torch.nn as nn
from torch.nn import Conv1d, ConvTranspose1d, LeakyReLU, Tanh, Conv2d, ConvTranspose2d, AvgPool1d, Upsample
from torch.nn.utils import weight_norm, spectral_norm

from src.utils import init_weights

from extra_utils import get_padding, get_padding_trans
from custom_layers import Conv1dRechanneled, Period1d, MelSpectrogram, AvgPool1dDilated
from custom_blocks import ResBlock, FusionBlock, FeatureBlock, ProcessedFeatureBlock, SumBlock, SubResBlock
from custom_discriminator import AllInOneBlock, AllInOneDiscriminator
from ensemble import Ensemble
from generator import Generator, Encoder, Decoder


def get_module_from_config(module_config, should_tag=False):
    tags = []
    if isinstance(module_config[-1], list) and isinstance(module_config[-1][0], str):
        tags = module_config[-1]
        module_config = module_config[:-1]
    if isinstance(module_config, list):
        module = nn.Sequential(*[
            get_module_from_config(sub_module, should_tag=should_tag) for sub_module in module_config
        ])
    elif len(module_config) == 1:
        module = get_no_params_module_from_config(*module_config, should_tag=should_tag)
    else:
        module = get_with_params_module_from_config(*module_config, should_tag=should_tag)

    if should_tag and len(tags) != 0:
        module.tags = tags
    return module


def get_no_params_module_from_config(module_name, should_tag=False):
    module = None
    if module_name == 'tanh':
        module = Tanh()

    return module


def get_with_params_module_from_config(module_name, module_parameters, should_tag):
    module = None
    if module_name == 'conv':
        chin, chout, kernel, stride, dilation, groups, norm = process_conv_params(*module_parameters)
        module = norm(
            Conv1d(chin, chout,
                   kernel_size=kernel,
                   stride=stride,
                   dilation=dilation,
                   groups=groups,
                   padding=get_padding(kernel, stride=stride, dilation=dilation)
                   )
        )
        module.apply(init_weights)
    elif module_name == 'conv_rech':
        chin, chout, kernel, stride, dilation, groups, norm = process_conv_params(*module_parameters)
        module = norm(
            Conv1dRechanneled(chin, chout,
                              kernel_size=kernel,
                              stride=stride,
                              dilation=dilation,
                              groups=groups,
                              padding=get_padding(kernel, stride=stride, dilation=dilation)
                              )
        )
        module.apply(init_weights)
    elif module_name == 'conv2':
        chin, chout, kernel, stride, dilation, groups, norm = process_conv_params(*module_parameters)
        module = norm(
            Conv2d(chin, chout,
                   kernel_size=kernel,
                   stride=stride,
                   dilation=dilation,
                   groups=groups,
                   padding=get_padding(kernel, stride=stride, dilation=dilation)
                   )
        )
        module.apply(init_weights)
    elif module_name == 'trans':
        chin, chout, kernel, stride, dilation, groups, norm = process_conv_params(*module_parameters)
        padding, output_padding = get_padding_trans(kernel, stride=stride, dilation=dilation)
        module = norm(
            ConvTranspose1d(chin, chout,
                            kernel_size=kernel,
                            stride=stride,
                            dilation=dilation,
                            groups=groups,
                            padding=padding,
                            output_padding=output_padding
                            )
        )
        module.apply(init_weights)
    elif module_name == 'trans2':
        chin, chout, kernel, stride, dilation, groups, norm = process_conv_params(*module_parameters)
        padding, output_padding = get_padding_trans(kernel, stride=stride, dilation=(1, 1))
        module = norm(
            ConvTranspose2d(chin, chout,
                            kernel_size=kernel,
                            stride=stride,
                            dilation=dilation,
                            groups=groups,
                            padding=padding,
                            output_padding=output_padding
                            )
        )
        module.apply(init_weights)
    elif module_name == 'up':
        stride, mode = module_parameters
        module = Upsample(scale_factor=stride, mode=mode)
    elif module_name == 'pool':
        kernel, stride = module_parameters
        module = AvgPool1d(kernel_size=kernel,
                           stride=stride,
                           padding=get_padding(kernel, stride=stride, dilation=1)
                           )
    elif module_name == 'poold':
        kernel, stride, dilation = module_parameters
        module = AvgPool1dDilated(kernel_size=kernel, stride=stride, dilation=dilation,
                                  padding=get_padding(kernel, stride=stride, dilation=1))
    elif module_name == 'period':
        period, padding_mode, padding_value = process_period_params(*module_parameters)
        module = Period1d(period=period,
                          padding_mode=padding_mode,
                          padding_value=padding_value)
    elif module_name == 'mel':
        module = MelSpectrogram(*module_parameters)
    elif module_name == 'fusion':
        modules_configs = module_parameters
        modules = nn.ModuleList([
            get_module_from_config(config, should_tag=should_tag) for config in modules_configs
        ])
        module = FusionBlock(modules)
    elif module_name == 'sum':
        modules_configs = module_parameters
        modules = nn.ModuleList([
            get_module_from_config(config, should_tag=should_tag) for config in modules_configs
        ])
        module = SumBlock(modules)
    elif module_name == 'res':
        module_config = module_parameters
        module = get_module_from_config(module_config, should_tag=should_tag)
        module = ResBlock(module)
    elif module_name == 'sub_res':
        module_config = module_parameters
        module = get_module_from_config(module_config, should_tag=should_tag)
        module = SubResBlock(module)
    elif module_name == 'lrelu':
        slope = module_parameters
        module = LeakyReLU(slope, inplace=False)
    elif module_name == 'fmap':
        module_config, tags = module_parameters
        module = get_module_from_config(module_config, should_tag=True)
        module = FeatureBlock(module, tags)
    elif module_name == 'pfmap':
        module_config, tags, feature_models_configs = module_parameters
        feature_models = nn.ModuleList([
            get_module_from_config(feature_model_config, should_tag=should_tag)
            for feature_model_config in feature_models_configs
        ])
        module = get_module_from_config(module_config, should_tag=True)
        module = ProcessedFeatureBlock(module, tags, feature_models)
    elif module_name == 'ensemble':
        blocks_configs = module_parameters
        blocks = nn.ModuleList([
            get_module_from_config(config, should_tag=should_tag) for config in blocks_configs
        ])
        module = Ensemble(blocks)
    elif module_name == 'all_in_one_block':
        before_block_config, raw_blocks_configs, after_block_config = module_parameters
        before_block = get_module_from_config(before_block_config, should_tag=should_tag)
        blocks = nn.ModuleList([
            get_module_from_config(config, should_tag=should_tag) for config in raw_blocks_configs
        ])
        after_block = get_module_from_config(after_block_config, should_tag=should_tag)
        module = AllInOneBlock(before_block, blocks, after_block)
    elif module_name == 'all_in_one_discriminator':
        before_block_config, blocks_configs, after_block_config = module_parameters
        before_block = get_module_from_config(before_block_config, should_tag=should_tag)
        blocks = nn.ModuleList([
            get_module_from_config(config, should_tag=should_tag) for config in blocks_configs
        ])
        after_block = get_module_from_config(after_block_config, should_tag=should_tag)
        module = AllInOneDiscriminator(before_block, blocks, after_block)
    elif module_name == 'encoder':
        vo_encoder_config, splitters_configs = module_parameters
        vo_encoder = get_module_from_config(vo_encoder_config, should_tag=should_tag)
        splitters = nn.ModuleList([
            get_module_from_config(config, should_tag=should_tag) for config in splitters_configs
        ])
        module = Encoder(vo_encoder, splitters)
    elif module_name == 'decoder':
        mergers_configs, vo_decoder_config = module_parameters
        mergers = nn.ModuleList([
            get_module_from_config(config, should_tag=should_tag) for config in mergers_configs
        ])
        vo_decoder = get_module_from_config(vo_decoder_config, should_tag=should_tag)
        module = Decoder(mergers, vo_decoder)
    elif module_name == 'generator':
        encoder_config, decoder_config = module_parameters
        encoder = get_module_from_config(encoder_config, should_tag=should_tag)
        decoder = get_module_from_config(decoder_config, should_tag=should_tag)
        module = Generator(encoder, decoder)

    return module


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
