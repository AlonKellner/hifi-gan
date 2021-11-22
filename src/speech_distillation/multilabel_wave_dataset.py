import json
import math
import os
import pickle
import random
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
import torch.utils.data

from augmentation.augmentation_methods import \
    NoiseAugmentor, RirAugmentor, CodecAugmentor, \
    LowpassAugmentor, HighpassAugmentor, ReverbAugmentor, \
    HilbertAugmentor
from complex_data_parser import get_path_by_glob, parse_complex_data
from src.meldataset import load_wav
from textgrid_parsing import parse_textgrid

PHI = (1 + math.sqrt(5))/2

MAX_WAV_VALUE = 32768.0

labels_to_use = ['speaker', 'sex', 'mic-brand']

timed_labels_to_use = ['phones']

label_groups = {
    'content': ['speaker', 'sex', 'phones'],
    'style': ['mic-brand']
}
augmentation_label_groups = {
    'content': [],
    'style': ['noise', 'rir', 'lowpass', 'highpass', 'reverb', 'codec', 'hilbert']
}


class MultilabelWaveDataset(torch.utils.data.Dataset):
    def __init__(self, base_dir, dir, name, config_path, segment_size, sampling_rate, embedding_size,
                 augmentation_config=None, disable_wavs=False, split=True, size=None,
                 fine_tuning=False, deterministic=False):
        self.base_dir = base_dir
        self.dir = dir
        self.name = name
        self.segment_size = segment_size
        self.embedding_size = embedding_size
        self.sampling_rate = sampling_rate
        self.split = split
        self.fine_tuning = fine_tuning
        self.size = size
        self.deterministic = deterministic
        self.random = random.Random()
        self.disable_wavs = disable_wavs
        self.should_augment = augmentation_config is not None
        if self.should_augment:
            self.aug_options = augmentation_config['options']
            self.aug_probs = augmentation_config['probs']
        print('Creating [{}] dataset:'.format(self.name))
        name_path = Path(os.path.join(dir, name))
        if not name_path.exists():
            os.mkdir(name_path)
        cache_path = Path(os.path.join(dir, name, 'labels_cache'))
        if not name_path.exists():
            os.mkdir(cache_path)
        self.files_with_labels = self.do_with_pickle_cache(lambda: self.get_files_with_labels(dir, config_path),
                                                           os.path.join(dir, name, 'files_with_labels.pickle'))
        self.label_options = self.do_with_pickle_cache(self.get_all_label_options,
                                                       os.path.join(dir, name, 'label_options.pickle'))
        for augmentation, augmentation_labels in self.aug_options.items():
            self.label_options[augmentation] = list({'none', *augmentation_labels})

        all_label_groups = {key: [*label_groups[key], *augmentation_label_groups[key]] for key in label_groups.keys()}
        self.label_option_groups = {
            key: {label: len(self.label_options[label]) for label in label_group}
            for key, label_group in all_label_groups.items()
        }

        if self.should_augment:
            self.aug_methods = {
                'noise': NoiseAugmentor(self.base_dir, self.label_options).augment,
                'rir': RirAugmentor(self.base_dir).augment,
                'reverb': ReverbAugmentor(self.sampling_rate).augment,
                'lowpass': LowpassAugmentor(self.sampling_rate).augment,
                'highpass': HighpassAugmentor(self.sampling_rate).augment,
                'codec': CodecAugmentor(self.sampling_rate).augment,
                'hilbert': HilbertAugmentor(self.sampling_rate).augment
            }

        if self.size is None:
            self.size = len(self.files_with_labels)
        print('Dataset [{}] is ready!\n'.format(self.name))

    @staticmethod
    def do_with_pickle_cache(func, pickle_path):
        pickle_path = Path(pickle_path)
        if pickle_path.exists():
            with open(pickle_path, 'rb') as pickle_file:
                result = pickle.load(pickle_file)
        else:
            result = func()
            with open(pickle_path, 'wb') as pickle_file:
                pickle.dump(result, pickle_file)
        return result

    def get_all_label_options(self):
        all_label_options = {}
        for col in labels_to_use:
            all_label_options[col] = set(self.files_with_labels[col].unique())

        with Pool(16) as pool:
            for label in timed_labels_to_use:
                all_label_options[label] = set()
            results = pool.map(self.get_uniques_timed_labels_by_index, range(len(self)))
        rows_to_remove = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                rows_to_remove.append(i)
            else:
                for label in timed_labels_to_use:
                    all_label_options[label].update(result[label])
        if len(rows_to_remove) > 0:
            self.files_with_labels = self.files_with_labels.drop(rows_to_remove).reset_index(drop=True)
            pickle_path = os.path.join(self.dir, self.name, 'files_with_labels.pickle')
            with open(pickle_path, 'wb') as pickle_file:
                pickle.dump(self.files_with_labels, pickle_file)
        all_label_options = {label: list(value) for label, value in all_label_options.items()}
        return all_label_options

    def get_uniques_timed_labels_by_index(self, i):
        try:
            labels, timed_labels = self.get_timed_labels(i)
            return self.get_unique_labels(timed_labels)
        except Exception as e:
            print('Item {} failed to get timed labels: [{}]'.format(i, e))
            return e

    def get_unique_labels(self, timed_labels):
        result = {}
        for label in timed_labels_to_use:
            result[label] = set(timed_labels[label]['text'].unique())
        return result

    def get_files_with_labels(self, main_dir, config_path):
        main_dir = Path(main_dir)
        subdir_list = [path for path in main_dir.glob('*/')]
        results = None
        for subdir in subdir_list:
            try:
                config_files = [path for path in subdir.glob(config_path)]
                for config_file in config_files:
                    config = config_file.read_text()
                    config_dict = json.loads(config)
                    print('Loading [{}]...'.format(config_dict['name']))
                    complex_data = parse_complex_data(subdir, config_dict['config'], config_dict['result'])
                    print('[{}] loaded successfully!'.format(config_dict['name']))
                    if results is None:
                        results = complex_data
                    else:
                        results = pd.concat([results, complex_data], axis=0, ignore_index=True)
            except Exception as e:
                print(e)
                print('Data config was not found or invalid, moving on.')
                continue

        return results

    def get_timed_labels(self, index):
        all_labels = self.files_with_labels.iloc[[index]].squeeze()
        labels = self.get_labels(index)
        timed_labels = parse_textgrid(all_labels['subdir'], all_labels['textgrid'])
        return labels, {key: value for key, value in timed_labels.items() if key in timed_labels_to_use}

    def get_labels(self, index):
        labels = self.files_with_labels[labels_to_use].iloc[[index]].squeeze()
        return labels

    def get_grouped_labels(self, index):
        labels = self.get_labels(index)
        grouped_labels = {group: labels.filter(group_labels).to_dict() for group, group_labels in label_groups.items()}
        return grouped_labels

    def __getitem__(self, index):
        if self.deterministic:
            self.random.seed(index)
        if self.size < len(self.files_with_labels):
            index = (int(len(self.files_with_labels) / PHI) * index) % len(self.files_with_labels)

        return self.get_augmented_item(index)

    def get_augmented_item(self, index):
        wav, wav_path, time_labels, grouped_labels = self.get_cut_item(index)
        if self.should_augment:
            wav, time_labels, grouped_labels = self.augment_item(wav, time_labels, grouped_labels)
        return wav, wav_path, time_labels, grouped_labels

    def get_pickle_label(self, index):
        return self.do_with_pickle_cache(
            lambda: self.get_fresh_label(index),
            os.path.join(self.dir, self.name, 'labels_cache', '{}.pickle'.format(index))
        )

    def get_fresh_label(self, index):
        labels, timed_labels = self.get_timed_labels(index)
        segmented_timed_labels = self.get_segmented_timed_labels(timed_labels)
        all_segmented_labels = self.add_segmented_labels(segmented_timed_labels, labels)
        segmented_tensor = self.convert_segmented_labels_to_tensor(all_segmented_labels, label_groups)
        return segmented_tensor

    def __len__(self):
        return min(len(self.files_with_labels), self.size)

    def get_segmented_timed_labels(self, timed_labels):
        return pd.concat(
            [
                self.get_segmented_timed_labels_for_single(label_name, timed_label)
                for label_name, timed_label in timed_labels.items()
            ],
            axis=1
        )

    def get_segmented_timed_labels_for_single(self, label_name, timed_label):
        result_rows = []
        time_interval = self.embedding_size / self.sampling_rate
        current_index = 0
        current_time = 0
        while current_index < len(timed_label):
            result_rows.append({label_name: timed_label.iloc[[current_index]].squeeze()['text']})
            current_time += time_interval
            if current_time > timed_label.iloc[[current_index]].squeeze()['end']:
                current_index += 1
        return pd.DataFrame(result_rows)

    def add_segmented_labels(self, segmented_timed_labels, labels):
        for col in labels.axes[0]:
            segmented_timed_labels[col] = labels[col]
        return segmented_timed_labels

    def convert_segmented_labels_to_tensor(self, all_segmented_labels, given_label_groups):
        all_tensors = {}
        for key, labels in given_label_groups.items():
            tensors = {}
            for col in labels:
                if col in all_segmented_labels:
                    index_tensor = torch.tensor(
                        all_segmented_labels[col].apply(lambda x: self.label_options[col].index(x)).tolist(),
                        dtype=torch.int64
                    )
                    tensors[col] = index_tensor
            all_tensors[key] = tensors
        return all_tensors

    def get_cut_wav(self, index):
        wav = self.get_wav(index)
        (length,) = wav.size()
        embedded_segment_size = self.segment_size // self.embedding_size
        embedded_length = length // self.embedding_size
        if embedded_length >= embedded_segment_size:
            max_embedded_start = embedded_length - embedded_segment_size
            embedded_start = self.random.randint(0, max_embedded_start)
            start = embedded_start * self.embedding_size

        if length >= self.segment_size:
            wav = wav[start:start + self.segment_size]
        else:
            wav = torch.nn.functional.pad(wav, (self.segment_size - length), 'constant')
        return wav

    def get_wav(self, index):
        wav_path = get_path_by_glob(self.dir, self.files_with_labels.iloc[[index]].squeeze()['wav'])
        if self.disable_wavs:
            return torch.zeros((self.segment_size,)), str(wav_path)
        audio, sampling_rate = load_wav(wav_path)

        if sampling_rate != self.sampling_rate:
            raise ValueError("{} SR doesn't match target {} SR".format(
                sampling_rate, self.sampling_rate))

        audio = torch.FloatTensor(audio)
        return audio.squeeze(0), str(wav_path)

    def get_cut_item(self, index):
        wav, wav_path = self.get_wav(index)
        pickle_label_groups = self.get_pickle_label(index)
        length = wav.size(0)
        embedded_segment_size = self.segment_size // self.embedding_size
        embedded_length = min(length // self.embedding_size,
                              next(iter(next(iter(pickle_label_groups.values())).values())).size(0))
        trimed_length = embedded_length * self.embedding_size
        trimed_start = 0
        if len(wav) > trimed_length:
            wav = wav[trimed_start:trimed_start + trimed_length]
        length = wav.size(0)
        # print(length, self.segment_size, embedded_length, embedded_segment_size)

        if length >= self.segment_size:
            max_embedded_start = embedded_length - embedded_segment_size
            embedded_start = self.random.randint(0, max_embedded_start)
            start = embedded_start * self.embedding_size
            # print('trim: ', start, embedded_start)
        else:
            embedded_padding = embedded_segment_size - embedded_length
            prefix_embedded_padding = self.random.randint(0, embedded_padding)
            postfix_embedded_padding = embedded_padding - prefix_embedded_padding
            padding = embedded_padding * self.embedding_size
            prefix_padding = prefix_embedded_padding * self.embedding_size
            postfix_padding = postfix_embedded_padding * self.embedding_size
            # print('pad: ', prefix_padding, postfix_padding, prefix_embedded_padding, postfix_embedded_padding, trimed_length)

        for key, group in pickle_label_groups.items():
            for label, label_item in group.items():
                label_item = label_item[0:embedded_length]
                if length >= self.segment_size:
                    cut_label_item = label_item[embedded_start:embedded_start + embedded_segment_size]
                else:
                    cut_label_item = torch.nn.functional.pad(label_item,
                                                             (prefix_embedded_padding, postfix_embedded_padding),
                                                             'constant')
                    # print(label, label_item.size(), cut_label_item.size())
                group[label] = cut_label_item

        if length >= self.segment_size:
            wav = wav[start:start + self.segment_size]
        else:
            wav = torch.nn.functional.pad(wav, (prefix_padding, postfix_padding), 'constant')

        grouped_labels = self.get_grouped_labels(index)
        return wav, wav_path, pickle_label_groups, grouped_labels

    def augment_item(self, cut_wav, cut_label, grouped_labels):
        options = self.aug_options
        probs = self.aug_probs
        methods = self.aug_methods
        (length,) = next(iter(next(iter(cut_label.values())).values())).size()
        augmented_wav = cut_wav
        augmented_label = pd.DataFrame(['none'] * length, columns=['none'])
        should_augment = probs['prob'] > self.random.random()
        for augmentation in options.keys():
            augmented_wav, augmented_label, value = self.augment_item_with(augmented_wav, augmented_label, cut_label,
                                                                           methods, options,
                                                                           probs, augmentation, should_augment)
            for section, current_label_groups in augmentation_label_groups.items():
                if augmentation in current_label_groups:
                    grouped_labels[section][augmentation] = value
        augmentation_tensors = self.convert_segmented_labels_to_tensor(augmented_label, augmentation_label_groups)
        for key in cut_label.keys():
            current_augmentation = augmentation_tensors[key]
            for label, value in current_augmentation.items():
                cut_label[key][label] = value
        return augmented_wav, cut_label, grouped_labels

    def augment_item_with(self, augmented_wav, augmented_label, cut_label, methods, options, probs, aug_type,
                          should=True):
        value = 'none'
        probs = probs['sub_probs'][aug_type]
        values = options[aug_type]
        aug_method = methods[aug_type]
        if should and probs['prob'] > self.random.random():
            value = self.random.choice(values)
            augmented_label, augmented_wav, value = aug_method(
                self.random,
                augmented_label,
                cut_label,
                augmented_wav,
                value,
                self.disable_wavs
            )
        augmented_label[aug_type] = value
        return augmented_wav, augmented_label, value
