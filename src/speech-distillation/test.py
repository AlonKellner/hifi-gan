import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import torch
from torchsummary import summary

from static_configs import get_static_generator_config, get_level5_model, get_level4_model, get_level3_model, \
        get_level2_model, get_level1_model, get_leveln_model
from configurable_module import get_module_from_config

# torch.cuda.manual_seed(1984)
# device = torch.device('cuda:{:d}'.format(0))
#
# initial_skip_ratio = 1
# generator = get_module_from_config(get_static_generator_config(1)).to(device)
model = get_module_from_config(('conv', (10, 10, 5)))
summary(model,
        input_size=(10, 10),
        batch_size=1,
        device='cpu')
