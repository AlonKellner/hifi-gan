from configurable_module import get_module_from_config
from torchsummary import summary
import torch
import torch.nn as nn
from torch.nn import LeakyReLU
from custom_wrappers import TagsWrapper
from static_configs import get_static_all_in_one_discriminator, \
        get_all_in_one_block_config, \
        get_static_generator_config
from custom_discriminator import AllInOneDiscriminator, AllInOneBlock
from custom_layers import Conv1dRechanneled, Period1d, MelSpectrogram, AvgPool1dDilated
from generator import Generator

test_config = ('fmap', [
        ('conv_rech', (1, 2, 3, 2)),
        ('conv_rech', (2, 3, 5, 3)),
        ('conv_rech', (3, 5, 9, 5)),
        ('conv_rech', (5, 8, 15, 8)),
        ('conv_rech', (8, 14, 25, 14)),
        ('conv_rech', (14, 20, 41, 20, 1, 2)),
        ('conv_rech', (20, 13, 25, 13, 1, 4)),
        ('conv_rech', (13, 8, 15, 8), True),
        ('conv_rech', (8, 5, 9, 5)),
        ('conv_rech', (5, 3, 5, 3)),
        ('conv_rech', (3, 2, 3, 2)),
        ('conv_rech', (2, 1, 3, 1)),
        ('conv', (1, 2, 3, 2)),
        ('conv', (2, 3, 5, 3), True),
        ('conv', (3, 5, 9, 5)),
        ('conv', (5, 8, 15, 8)),
        ('conv', (8, 13, 25, 13)),
        ('conv', (13, 21, 41, 21)),
        ('trans', (21, 13, 25, 13)),
        ('trans', (13, 8, 15, 8)),
        ('trans', (8, 5, 9, 5)),
        ('trans', (5, 3, 5, 3)),
        ('trans', (3, 2, 3, 2)),
        ('trans', (2, 1, 3, 1)),
        ('period', (5,)),
        ('period', (3,)),
        ('period', (2,)),
    ])


# device = torch.device('cuda:{:d}'.format(0))
# generator = Generator(get_static_generator_config()).to(device)
# summary(generator,
#         input_size=(1, 21840),
#         batch_size=2)
# discriminator = get_module_from_config(get_static_all_in_one_discriminator(8)).to(device)
# summary(discriminator,
#         input_size=(1, 21840),
#         batch_size=2)
layer = Period1d(period=5)
values = torch.linspace(5, 6, 10).reshape(1, 1, 10)
zeros = torch.zeros(1, 2, 10)
mix = torch.cat([values, zeros], dim=1)
print(mix)
print(layer(mix))
