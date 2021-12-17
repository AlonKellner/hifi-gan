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

# torch.cuda.manual_seed(1984)
# device = torch.device('cuda:{:d}'.format(0))
#
# initial_skip_ratio = 1

# model = get_module_from_config(('conv', (10, 10, 5)))
# model = get_module_from_config(get_res_block_config(16, 33))
# summary(model,
#         input_size=(16, 100),
#         batch_size=1,
#         device='cpu')
# with open('config/config.json') as f:
#     data = f.read()
#
# config_dict = json.loads(data)
# h = AttrDict(config_dict)
#
# train_dataset = MultilabelWaveDataset(
#         base_dir='/datasets',
#         dir='/datasets/training_audio',
#         name='train',
#         config_path='**/train_data_config/*.json',
#         segment_length=h.segment_length,
#         sampling_rate=h.sampling_rate,
#         embedding_size=h.embedding_size,
#         augmentation_config=config_dict['augmentation']
#     )
#
# example_item = train_dataset.label_option_groups
#
# keepers = generate_keepers_by_example(
#         h.embedding_size, example_item
#     )
#
# for key, keeper in keepers.items():
#     print(f'{key} keeper:')
#     summary(keeper,
#             input_size=(h.embedding_size, h.segment_length // h.embedding_size),
#             batch_size=h.batch_size,
#             device='cpu')
#
#
# result = keepers['style'](torch.randn((1, h.embedding_size, h.segment_length // h.embedding_size)))
# print(result['style']['mic-brand'][0, :, 0].sum())
#
# size = (3, 5)
# dim = 0
# ground_truth = torch.randint(low=0, high=size[dim], size=(*size[:dim], *size[dim+1:]))
# x = torch.randint(low=0, high=size[dim], size=(*size[:dim], *size[dim+1:]))
# x = F.one_hot(x, size[dim]).transpose(-1, dim)
# # logits = torch.randn(size)
# # x = torch.softmax(logits, dim=0)
# target = torch.softmax(torch.randn(size), dim=0)
# value = bias_corrected_cross_entropy_loss(x, target, ground_truth, dim=dim)
# print(value)
# #
# discriminator = get_module_from_config(get_static_all_in_one_discriminator(8))
# summary(discriminator,
#         input_size=(1, h.segment_length),
#         batch_size=h.batch_size,
#         device='cpu')
#
# generator = get_module_from_config(
#     get_static_generator_config(
#         initial_skip_ratio=h.initial_skip_ratio,
#         expansion_size=h.expansion_size
#     )
# )
# summary(generator,
#         input_size=(1, h.segment_length),
#         batch_size=h.batch_size,
#         device='cpu')


# subdir = '/datasets/LibriSpeech'
# textgrid_pattern = '**/librispeech_alignments/test-clean/6930/76324/6930-76324-0017.TextGrid'
#
#
# result = parse_textgrid(subdir, textgrid_pattern)
# print(result)
#
# train_dataset = MultilabelWaveDataset(
#     base_dir='/datasets',
#     dir='/datasets/training_audio',
#     name='train',
#     config_path='**/train_data_config/*.json',
#     segment_length=h.segment_length,
#     sampling_rate=h.sampling_rate,
#     embedding_size=h.embedding_size,
#     augmentation_config=config_dict['augmentation']
# )
#
# validation_dataset = MultilabelWaveDataset(
#     base_dir='/datasets',
#     dir='/datasets/training_audio',
#     name='train',
#     config_path='**/train_data_config/*.json',
#     segment_length=h.validation_segment_length,
#     sampling_rate=h.sampling_rate,
#     embedding_size=h.embedding_size,
#     augmentation_config=config_dict['augmentation'],
#     deterministic=True,
#     size=h.validation_amount
# )
#
# test_dataset = MultilabelWaveDataset(
#     base_dir='/datasets',
#     dir='/datasets/training_audio',
#     name='test',
#     config_path='**/test_data_config/*.json',
#     segment_length=h.validation_segment_length,
#     sampling_rate=h.sampling_rate,
#     embedding_size=h.embedding_size,
#     deterministic=True,
#     augmentation_config=config_dict['augmentation']
# )
# torch.set_printoptions(profile='full')
# print(train_dataset.get_fresh_label(0))
# print(test_dataset.get_fresh_label(0))

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
# with Pool(16) as pool:
#     pool.map(train_dataset.create_pickle_label, range(len(train_dataset)))
#
# with Pool(16) as pool:
#     pool.map(test_dataset.create_pickle_label, range(len(test_dataset)))
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

batch_size = 6
mixing_size = 7
total_size = batch_size + mixing_size
cycles = (batch_size, *calculate_cycles(batch_size, mixing_size))
a = torch.randn((batch_size, 3))
u = expand(a, sum(cycles), dim=0)
r = mix(u, cycles, dim=0)
print(r.size())
print(u.size())

