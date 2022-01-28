import pickle
import random
import time
import warnings
import json
from src.env import AttrDict, build_env
from pathlib import Path

from src.speech_distillation.custom_losses import bias_corrected_cross_entropy_loss
from src.speech_distillation.cycle_calculator import calculate_cycles
from src.speech_distillation.embedding_classifiers.embedding_classifiers_static_configs import \
    generate_keepers_by_example
from src.speech_distillation.label_bias_sniffer import generate_sniffers_by_example
from src.speech_distillation.lightning_model import create_datasets
from src.speech_distillation.tensor_utils import mix, expand
from textgrid_parsing import parse_textgrid
from multilabel_wave_dataset import MultilabelWaveDataset
from multiprocessing import Pool
from src.meldataset import save_wav
warnings.simplefilter(action='ignore', category=FutureWarning)

import torch
from torch.nn import functional as F
from torchsummary import summary

from static_configs import get_generator_configs, get_last_level_model, \
    get_leveln_model, get_block_config, get_discriminator_config
from configurable_module import get_module_from_config

with open('config/config.json') as f:
    data = f.read()

config = json.loads(data)

datasets = create_datasets(
    config['loops'], config['data'], config['augmentation'], config['sampling_rate'], config['embedding_size']
)
train_dataset = datasets['train']['dataset']
test_dataset = datasets['test']['dataset']

torch.set_printoptions(profile='full')
print(train_dataset.get_fresh_label(0))
print(test_dataset.get_fresh_label(0))


# validation_dataset = datasets['validation']['dataset']
# print(len(validation_dataset))
# for i, item in enumerate(validation_dataset):
#     if i % 100 == 0:
#         print(i)
# print('validation finished!')

# print(len(train_dataset))
# for i, item in enumerate(train_dataset):
#     if i % 100 == 0:
#         print(i)
# print('train finished!')
#
# print(len(test_dataset))
# for i, item in enumerate(test_dataset):
#     if i % 100 == 0:
#         print(i)
# print('test finished!')

#
with Pool(16) as pool:
    pool.map(train_dataset.create_pickle_label, range(len(train_dataset)))

with Pool(16) as pool:
    pool.map(test_dataset.create_pickle_label, range(len(test_dataset)))
# index = 0
# print('train')
# for i, item in enumerate(train_dataset):
#     index = i
#     wav, wav_path, time_labels, str_labels = item
#     str_labels['path'] = str(wav_path)
#     save_wav('/mount/wavs/train/{}.wav'.format(i), wav.unsqueeze(0), h.sampling_rate)
#     with open('/mount/wavs/train/{}.json'.format(i), 'w') as file:
#         json_labels = json.dumps(str_labels, indent=4, sort_keys=True)
#         file.write(json_labels)
#     if wav.size(0) % time_labels['content']['speaker'].size(0) != 0:
#         print('ERROR', i)
#         print(wav.size())
#         print(time_labels['content']['speaker'].size())
#     if i % 10000 == 0:
#         print(i)
# print(index)
# print('test')
# for i, item in enumerate(test_dataset):
#     index = i
#     wav, wav_path, time_labels, str_labels = item
#     str_labels['path'] = str(wav_path)
#     save_wav('/mount/wavs/test/{}.wav'.format(i), wav.unsqueeze(0), h.sampling_rate)
#     with open('/mount/wavs/test/{}.json'.format(i), 'w') as file:
#         json_labels = json.dumps(str_labels, indent=4, sort_keys=True)
#         file.write(json_labels)
#     if wav.size(0) % time_labels['content']['speaker'].size(0) != 0:
#         print('ERROR', i)
#         print(wav.size())
#         print(time_labels['content']['speaker'].size())
#     if i % 10000 == 0:
#         print(i)
# print(index)
# print(dataset)

# batch_size = 6
# mixing_size = 7
# total_size = batch_size + mixing_size
# cycles = (batch_size, *calculate_cycles(batch_size, mixing_size))
# a = torch.randn((batch_size, 3))
# u = expand(a, sum(cycles), dim=0)
# r = mix(u, cycles, dim=0)
# print(r.size())
# print(u.size())

print('done!')

