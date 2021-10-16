import os
import pickle
import random

import pandas as pd
import torch
import torch.utils.data
import torch.nn.functional as F
from librosa.util import normalize
from pathlib import Path
import re
import json
from csv import reader
from multiprocessing import Pool

from src.meldataset import load_wav
from complex_data_parser import get_path_by_glob, parse_complex_data
from textgrid_parsing import parse_textgrid

from augmentation.augmentation_methods import noise_augmentation, rir_augmentation, sox_augmentation

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


class MultilabelDataset(torch.utils.data.Dataset):
    def __init__(self, dir, name, config_path, segment_size, sampling_rate, embedding_size, augmentation_config, split=True, n_cache_reuse=1,
                 fine_tuning=False, deterministic=False):
        self.dir = dir
        self.name = name
        self.segment_size = segment_size
        self.embedding_size = embedding_size
        self.sampling_rate = sampling_rate
        self.split = split
        self.n_cache_reuse = n_cache_reuse
        self.fine_tuning = fine_tuning
        self._cache_ref_count = 0
        self.deterministic = deterministic
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
        all_label_groups = {key: [*label_groups[key], *augmentation_label_groups[key]] for key in label_groups.keys()}
        self.label_option_groups = {
            key: {label: len(self.label_options[label]) for label in label_group}
            for key, label_group in all_label_groups.items()
        }
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

        for augmentation, augmentation_labels in self.aug_options.items():
            all_label_options[augmentation] = {'none', *augmentation_labels}

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
        labels = self.files_with_labels[labels_to_use].iloc[[index]].squeeze()
        timed_labels = parse_textgrid(all_labels['subdir'], all_labels['textgrid'])
        return labels, {key: value for key, value in timed_labels.items() if key in timed_labels_to_use}

    def __getitem__(self, index):
        return self.get_augmented_item(index)

    def get_augmented_item(self, index):
        cut_item = self.get_cut_item(index)
        return self.augment_item(cut_item)

    def get_pickle_item(self, index):
        return self.do_with_pickle_cache(
            lambda: self.get_fresh_item(index),
            os.path.join(self.dir, self.name, 'labels_cache', '{}.pickle'.format(index))
        )

    def get_fresh_item(self, index):
        labels, timed_labels = self.get_timed_labels(index)
        segmented_timed_labels = self.get_segmented_timed_labels(timed_labels)
        all_segmented_labels = self.add_segmented_labels(segmented_timed_labels, labels)
        segmented_tensor = self.convert_segmented_labels_to_tensor(all_segmented_labels, label_groups)
        return segmented_tensor

    def __len__(self):
        return len(self.files_with_labels)

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

    def get_cut_item(self, index):
        pickle_label_groups = self.get_pickle_item(index)
        embedded_segment_size = self.segment_size // self.embedding_size
        (embedded_length,) = next(iter(next(iter(pickle_label_groups.values())).values())).size()
        if embedded_length >= embedded_segment_size:
            max_embedded_start = embedded_length - embedded_segment_size
            embedded_start = 0 if self.deterministic else random.randint(0, max_embedded_start)

        for key, group in pickle_label_groups.items():
            for label, label_item in group.items():
                if embedded_length >= embedded_segment_size:
                    cut_label_item = label_item[embedded_start:embedded_start + embedded_segment_size]
                else:
                    cut_label_item = torch.nn.functional.pad(label_item, (0, embedded_segment_size - embedded_length), 'constant')
                group[label] = cut_label_item
        return pickle_label_groups

    def augment_item(self, cut_item):
        (length, ) = next(iter(next(iter(cut_item.values())).values())).size()
        augmented_item = pd.DataFrame(['none'] * length, columns=['none'])
        should_augment = self.aug_probs['prob'] > random.random()
        augmented_item = self.augment_with(augmented_item, self.aug_probs, 'noise', should_augment)
        augmented_item = self.augment_with(augmented_item, self.aug_probs, 'rir', should_augment)
        augmented_item = self.augment_with_sox(augmented_item, self.aug_probs, should_augment)
        augmentation_tensors = self.convert_segmented_labels_to_tensor(augmented_item, augmentation_label_groups)
        for key in cut_item.keys():
            current_augmentation = augmentation_tensors[key]
            for label, value in current_augmentation.items():
                cut_item[key][label] = value
        return cut_item

    def augment_with(self, augmented_item, probs, aug_type, should=True):
        value = 'none'
        probs = probs['sub_probs'][aug_type]
        values = self.aug_options[aug_type]
        if should and probs['prob'] > random.random():
            value = values[random.randrange(0, len(values))]
        augmented_item[aug_type] = value
        return augmented_item

    def augment_with_sox(self, augmented_item, probs, should_augment):
        probs = probs['sub_probs']['sox']
        should_augment = should_augment and probs['prob'] > random.random()
        augmented_item = self.augment_with(augmented_item, probs, 'lowpass', should_augment)
        augmented_item = self.augment_with(augmented_item, probs, 'highpass', should_augment)
        augmented_item = self.augment_with(augmented_item, probs, 'reverb', should_augment)
        augmented_item = self.augment_with(augmented_item, probs, 'codec', should_augment)
        augmented_item = self.augment_with(augmented_item, probs, 'hilbert', should_augment)
        return augmented_item


class MultilabelWaveDataset(torch.utils.data.Dataset):
    def __init__(self, dir, name, config_path, segment_size, sampling_rate, embedding_size, augmentation_config, split=True, n_cache_reuse=1,
                 fine_tuning=False, deterministic=False):
        self.dir = dir
        self.name = name
        self.segment_size = segment_size
        self.embedding_size = embedding_size
        self.sampling_rate = sampling_rate
        self.split = split
        self.n_cache_reuse = n_cache_reuse
        self.fine_tuning = fine_tuning
        self._cache_ref_count = 0
        self.deterministic = deterministic
        self.aug_options = augmentation_config['options']
        self.aug_probs = augmentation_config['probs']
        self.aug_methods = {
            'noise': noise_augmentation,
            'rir': rir_augmentation,
            'sox': {
                'lowpass': sox_augmentation,
                'highpass': sox_augmentation,
                'reverb': sox_augmentation,
                'codec': sox_augmentation,
                'hilbert': sox_augmentation,
            }
        }
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
        all_label_groups = {key: [*label_groups[key], *augmentation_label_groups[key]] for key in label_groups.keys()}
        self.label_option_groups = {
            key: {label: len(self.label_options[label]) for label in label_group}
            for key, label_group in all_label_groups.items()
        }
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

        for augmentation, augmentation_labels in self.aug_options.items():
            all_label_options[augmentation] = {'none', *augmentation_labels}

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
        labels = self.files_with_labels[labels_to_use].iloc[[index]].squeeze()
        timed_labels = parse_textgrid(all_labels['subdir'], all_labels['textgrid'])
        return labels, {key: value for key, value in timed_labels.items() if key in timed_labels_to_use}

    def __getitem__(self, index):
        return self.get_augmented_item(index)

    def get_augmented_label(self, index):
        cut_label = self.get_cut_label(index)
        return self.augment_label(cut_label)

    def get_augmented_item(self, index):
        cut_wav, cut_label = self.get_cut_item(index)
        return self.augment_item(cut_wav, cut_label)

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
        return len(self.files_with_labels)

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

    def get_cut_label(self, index):
        pickle_label_groups = self.get_pickle_label(index)
        embedded_segment_size = self.segment_size // self.embedding_size
        (embedded_length,) = next(iter(next(iter(pickle_label_groups.values())).values())).size()
        if embedded_length >= embedded_segment_size:
            max_embedded_start = embedded_length - embedded_segment_size
            embedded_start = 0 if self.deterministic else random.randint(0, max_embedded_start)

        for key, group in pickle_label_groups.items():
            for label, label_item in group.items():
                if embedded_length >= self.segment_size:
                    cut_label_item = label_item[embedded_start:embedded_start + embedded_segment_size]
                else:
                    cut_label_item = torch.nn.functional.pad(label_item, (0, embedded_segment_size - embedded_length), 'constant')
                group[label] = cut_label_item
        return pickle_label_groups

    def augment_label(self, cut_label):
        (length, ) = next(iter(next(iter(cut_label.values())).values())).size()
        augmented_label = pd.DataFrame(['none'] * length, columns=['none'])
        should_augment = self.aug_probs['prob'] > random.random()
        augmented_label = self.augment_label_with(augmented_label, self.aug_probs, 'noise', should_augment)
        augmented_label = self.augment_label_with(augmented_label, self.aug_probs, 'rir', should_augment)
        augmented_label = self.augment_label_with_sox(augmented_label, self.aug_probs, should_augment)
        augmentation_tensors = self.convert_segmented_labels_to_tensor(augmented_label, augmentation_label_groups)
        for key in cut_label.keys():
            current_augmentation = augmentation_tensors[key]
            for label, value in current_augmentation.items():
                cut_label[key][label] = value
        return cut_label

    def augment_label_with(self, augmented_label, probs, aug_type, should=True):
        value = 'none'
        probs = probs['sub_probs'][aug_type]
        values = self.aug_options[aug_type]
        if should and probs['prob'] > random.random():
            value = values[random.randrange(0, len(values))]
        augmented_label[aug_type] = value
        return augmented_label

    def augment_label_with_sox(self, augmented_label, probs, should_augment):
        probs = probs.sub_probs['sox']
        should_augment = should_augment and probs.prob > random.random()
        augmented_label = self.augment_label_with(augmented_label, probs, 'lowpass', should_augment)
        augmented_label = self.augment_label_with(augmented_label, probs, 'highpass', should_augment)
        augmented_label = self.augment_label_with(augmented_label, probs, 'reverb', should_augment)
        augmented_label = self.augment_label_with(augmented_label, probs, 'codec', should_augment)
        augmented_label = self.augment_label_with(augmented_label, probs, 'hilbert', should_augment)
        return augmented_label

    # def __init__(self, dir, config_path, segment_size, sampling_rate, split=True, n_cache_reuse=1,
    #              fine_tuning=False, deterministic=False):
    #     self.segment_size = segment_size
    #     self.sampling_rate = sampling_rate
    #     self.split = split
    #     self.n_cache_reuse = n_cache_reuse
    #     self.fine_tuning = fine_tuning
    #     self._cache_ref_count = 0
    #     self.deterministic = deterministic
    #     self.files_with_labels = self.get_files_with_labels(dir, config_path)

    def get_cut_wav(self, index):
        wav = self.get_wav(index)
        (length,) = wav.size()
        embedded_segment_size = self.segment_size // self.embedding_size
        embedded_length = length // self.embedding_size
        if embedded_length >= embedded_segment_size:
            max_embedded_start = embedded_length - embedded_segment_size
            embedded_start = 0 if self.deterministic else random.randint(0, max_embedded_start)
            start = embedded_start * self.embedding_size

        if length >= self.segment_size:
            wav = wav[start:start + self.segment_size]
        else:
            wav = torch.nn.functional.pad(wav, (self.segment_size - length), 'constant')
        return wav

    def get_wav(self, index):
        wav_path = get_path_by_glob(self.dir, self.files_with_labels.iloc[[index]].squeeze()['wav'])
        if self._cache_ref_count == 0:
            audio, sampling_rate = load_wav(wav_path)
            self.cached_wav = audio
            if sampling_rate != self.sampling_rate:
                raise ValueError("{} SR doesn't match target {} SR".format(
                    sampling_rate, self.sampling_rate))
            self._cache_ref_count = self.n_cache_reuse
        else:
            audio = self.cached_wav
            self._cache_ref_count -= 1

        audio = torch.FloatTensor(audio)
        return audio.squeeze(0)

    def get_cut_item(self, index):
        wav = self.get_wav(index)
        pickle_label_groups = self.get_pickle_label(index)
        length = wav.size(0)
        embedded_segment_size = self.segment_size // self.embedding_size
        embedded_length = min(length // self.embedding_size, next(iter(next(iter(pickle_label_groups.values())).values())).size(0))
        trimed_length = embedded_length * self.embedding_size
        trimed_start = 0
        if len(wav) > trimed_length:
            wav = wav[trimed_start:trimed_start+trimed_length]
        length = wav.size(0)
        # print(length, self.segment_size, embedded_length, embedded_segment_size)

        if length >= self.segment_size:
            max_embedded_start = embedded_length - embedded_segment_size
            embedded_start = 0 if self.deterministic else random.randint(0, max_embedded_start)
            start = embedded_start * self.embedding_size
            # print('trim: ', start, embedded_start)
        else:
            embedded_padding = embedded_segment_size - embedded_length
            prefix_embedded_padding = 0 if self.deterministic else random.randint(0, embedded_padding)
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
                    cut_label_item = torch.nn.functional.pad(label_item, (prefix_embedded_padding, postfix_embedded_padding), 'constant')
                    # print(label, label_item.size(), cut_label_item.size())
                group[label] = cut_label_item

        if length >= self.segment_size:
            wav = wav[start:start + self.segment_size]
        else:
            wav = torch.nn.functional.pad(wav, (prefix_padding, postfix_padding), 'constant')

        return wav, pickle_label_groups

    def augment_item(self, cut_wav, cut_label):
        options = self.aug_options
        probs = self.aug_probs
        methods = self.aug_methods
        (length, ) = next(iter(next(iter(cut_label.values())).values())).size()
        augmented_wav = cut_wav
        augmented_label = pd.DataFrame(['none'] * length, columns=['none'])
        should_augment = probs['prob'] > random.random()
        augmented_wav, augmented_label = self.augment_item_with(augmented_wav, augmented_label, methods, options, probs, 'noise', should_augment)
        augmented_wav, augmented_label = self.augment_item_with(augmented_wav, augmented_label, methods, options, probs, 'rir', should_augment)
        augmented_wav, augmented_label = self.augment_item_with_sox(augmented_wav, augmented_label, methods, options, probs, should_augment)
        augmentation_tensors = self.convert_segmented_labels_to_tensor(augmented_label, augmentation_label_groups)
        for key in cut_label.keys():
            current_augmentation = augmentation_tensors[key]
            for label, value in current_augmentation.items():
                cut_label[key][label] = value
        return augmented_wav, cut_label

    def augment_item_with(self, augmented_wav, augmented_label, methods, options, probs, aug_type, should=True):
        value = 'none'
        probs = probs['sub_probs'][aug_type]
        values = options[aug_type]
        aug_method = methods[aug_type]
        if should and probs['prob'] > random.random():
            value = values[random.randrange(0, len(values))]
            augmented_wav = aug_method(augmented_wav)
        augmented_label[aug_type] = value
        return augmented_wav, augmented_label

    def augment_item_with_sox(self, augmented_wav, augmented_label, methods, options, probs, should_augment):
        probs = probs['sub_probs']['sox']
        options = options['sox']
        methods = methods['sox']
        should_augment = should_augment and probs['prob'] > random.random()
        augmented_wav, augmented_label = self.augment_item_with(augmented_wav, augmented_label, methods, options, probs, 'lowpass', should_augment)
        augmented_wav, augmented_label = self.augment_item_with(augmented_wav, augmented_label, methods, options, probs, 'highpass', should_augment)
        augmented_wav, augmented_label = self.augment_item_with(augmented_wav, augmented_label, methods, options, probs, 'reverb', should_augment)
        augmented_wav, augmented_label = self.augment_item_with(augmented_wav, augmented_label, methods, options, probs, 'codec', should_augment)
        augmented_wav, augmented_label = self.augment_item_with(augmented_wav, augmented_label, methods, options, probs, 'hilbert', should_augment)
        return augmented_wav, augmented_label
