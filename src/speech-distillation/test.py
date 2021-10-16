import warnings
import json
from src.env import AttrDict, build_env
from pathlib import Path
from textgrid_parsing import parse_textgrid
from multilabel_wave_dataset import MultilabelDataset, MultilabelWaveDataset
from multiprocessing import Pool
warnings.simplefilter(action='ignore', category=FutureWarning)

import torch
from torchsummary import summary

from static_configs import get_static_generator_config, get_level5_model, \
    get_leveln_model, get_res_block_config, get_static_all_in_one_discriminator
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
with open('config/config_none.json') as f:
    data = f.read()

json_config = json.loads(data)
h = AttrDict(json_config)
#
# discriminator = get_module_from_config(get_static_all_in_one_discriminator(8))
# summary(discriminator,
#         input_size=(1, h.segment_size),
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
#         input_size=(1, h.segment_size),
#         batch_size=h.batch_size,
#         device='cpu')


# subdir = '/datasets/LibriSpeech'
# textgrid_pattern = '**/librispeech_alignments/test-clean/6930/76324/6930-76324-0017.TextGrid'
#
#
# result = parse_textgrid(subdir, textgrid_pattern)
# print(result)

train_dataset = MultilabelWaveDataset(
    dir='/datasets/training_audio',
    name='train',
    config_path='**/train_data_config/*.json',
    segment_size=h.segment_size,
    sampling_rate=h.sampling_rate,
    embedding_size=3 * 7 * 13,
    augmentation_config=json_config['augmentation']
)

test_dataset = MultilabelWaveDataset(
    dir='/datasets/training_audio',
    name='test',
    config_path='**/test_data_config/*.json',
    segment_size=h.segment_size,
    sampling_rate=h.sampling_rate,
    embedding_size=3 * 7 * 13,
    augmentation_config=json_config['augmentation']
)

# torch.set_printoptions(profile='full')
# print(dataset.get_pickle_item(0))

# with Pool(16) as pool:
#     results = pool.map(train_dataset.get_pickle_item, range(len(train_dataset)))

# with Pool(16) as pool:
#     results = pool.map(test_dataset.get_pickle_item, range(len(test_dataset)))
index = 0
print('train')
for i, item in enumerate(train_dataset):
    index = i
    wav, label = item
    if wav.size(0) % label['content']['speaker'].size(0) != 0:
        print('ERROR', i)
        print(wav.size())
        print(label['content']['speaker'].size())
    if i % 10000 == 0:
        print(i)
print(index)
print('test')
for i, item in enumerate(test_dataset):
    index = i
    wav, label = item
    if wav.size(0) % label['content']['speaker'].size(0) != 0:
        print('ERROR', i)
        print(wav.size())
        print(label['content']['speaker'].size())
    if i % 10000 == 0:
        print(i)
print(index)
# print(dataset)
