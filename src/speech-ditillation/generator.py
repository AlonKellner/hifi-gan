import torch
import torch.nn as nn
from torch.nn import Conv1d, ConvTranspose1d, LeakyReLU, Tanh
from torch.nn.utils import weight_norm, remove_weight_norm

from blocks import Conv1dResBlockFusion
from src.utils import init_weights

LRELU_SLOPE = 0.1


def get_static_generator():
    common_res_blocks = \
        [
            [
                [(5, 1), (5, 1)],
                [(5, 3), (5, 1)],
                [(5, 5), (5, 1)]
            ],
            [
                [(8, 1), (8, 1)],
                [(8, 3), (8, 1)],
                [(8, 5), (8, 1)]
            ],
            [
                [(13, 1), (13, 1)],
                [(13, 3), (13, 1)],
                [(13, 5), (13, 1)]
            ]
        ]

    model_config = [
        ("conv", (1, 32, 7, 1)),
        ("res_fusion", (32, common_res_blocks)),
        ("lrelu", LRELU_SLOPE),
        ("conv", (32, 64, 15, 2)),
        ("res_fusion", (64, common_res_blocks)),
        ("lrelu", LRELU_SLOPE),
        ("conv", (64, 128, 31, 2)),
        ("res_fusion", (128, common_res_blocks)),
        ("lrelu", LRELU_SLOPE),
        ("conv", (128, 128, 31, 1)),
        ("res_fusion", (128, common_res_blocks)),
        ("lrelu", LRELU_SLOPE),
        ("trans", (128, 64, 31, 2)),
        ("res_fusion", (64, common_res_blocks)),
        ("lrelu", LRELU_SLOPE),
        ("trans", (64, 32, 15, 2)),
        ("res_fusion", (32, common_res_blocks)),
        ("lrelu", LRELU_SLOPE),
        ("trans", (32, 1, 7, 1)),
        ("tanh",)
    ]
    return ConfigurableModel(model_config)


class ConfigurableModel(torch.nn.Module):
    def __init__(self, model_config):
        super(ConfigurableModel, self).__init__()

        self.model = self.get_model_from_config(model_config)

    def get_model_from_config(self, model_config):
        return nn.Sequential(*[
            self.get_block_from_config(block_config) for block_config in model_config
        ])

    @staticmethod
    def get_block_from_config(block_config):
        if len(block_config) == 1:
            block_name = block_config[0]
            if block_name == "tanh":
                return Tanh()
        if len(block_config) == 2:
            block_name, block_parameters = block_config
            if block_name == "conv":
                chin, chout, kernel, stride = block_parameters
                layer = weight_norm(Conv1d(chin, chout, kernel, stride))
                layer.apply(init_weights)
                return layer
            elif block_name == "trans":
                chin, chout, kernel, stride = block_parameters
                layer = weight_norm(ConvTranspose1d(chin, chout, kernel, stride))
                layer.apply(init_weights)
                return layer
            elif block_name == "res_fusion":
                chin, blocks = block_parameters
                return weight_norm(Conv1dResBlockFusion(chin, blocks))
            elif block_name == "lrelu":
                slope = block_parameters[0]
                return LeakyReLU(slope)

    def forward(self, x):
        return self.model(x)

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for module in self.model.modules():
            if hasattr(module, 'remove_weight_norm'):
                module.remove_weight_norm()
            else:
                remove_weight_norm(module)
