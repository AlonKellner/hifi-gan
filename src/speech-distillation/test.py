from configurable_module import ConfigurableModule
from torchsummary import summary
import torch


test_config = [
        ('conv_rech', (1, 2, 3, 2)),
        ('conv_rech', (2, 3, 5, 3)),
        ('conv_rech', (3, 5, 9, 5)),
        ('conv_rech', (5, 8, 15, 8)),
        ('conv_rech', (8, 14, 25, 14)),
        ('conv_rech', (14, 20, 41, 20, 1, 2)),
        ('conv_rech', (20, 13, 25, 13, 1, 4)),
        ('conv_rech', (13, 8, 15, 8)),
        ('conv_rech', (8, 5, 9, 5)),
        ('conv_rech', (5, 3, 5, 3)),
        ('conv_rech', (3, 2, 3, 2)),
        ('conv_rech', (2, 1, 3, 1)),
        ('conv', (1, 2, 3, 2)),
        ('conv', (2, 3, 5, 3)),
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
    ]


device = torch.device('cuda:{:d}'.format(0))
model = ConfigurableModule(test_config).to(device)

summary(model, input_size=(1, 10920))
