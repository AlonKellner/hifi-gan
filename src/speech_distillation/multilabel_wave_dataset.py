from collections import OrderedDict
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

PHI = (1 + math.sqrt(5)) / 2

MAX_WAV_VALUE = 32768.0

labels_to_use = ['speaker', 'sex', 'mic-brand']

sad_based_labels = ['sex', 'speaker']

timed_labels_to_use = ['phones', 'sex', 'speaker', 'sad']

label_groups = {
    'content': ['speaker', 'sex', 'phones', 'sad'],
    'style': ['mic-brand']
}
augmentation_label_groups = {
    'content': [],
    'style': ['noise', 'rir', 'lowpass', 'highpass', 'reverb', 'codec', 'hilbert']
}


class MultilabelWaveDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, aug_dir, cache_dir, name, source, segment_length, sampling_rate, embedding_size,
                 augmentation_config=None, disable_wavs=False, split=True, size=None,
                 fine_tuning=False, deterministic=False):
        self.data_dir = data_dir
        self.aug_dir = aug_dir
        self.cache_dir = cache_dir
        self.name = name
        self.source = source
        self.segment_length = segment_length
        self.embedding_size = embedding_size
        self.sampling_rate = sampling_rate
        self.split = split
        self.fine_tuning = fine_tuning
        self.size = size
        self.deterministic = deterministic
        self.disable_wavs = disable_wavs
        self.should_augment = augmentation_config is not None
        if self.should_augment:
            self.aug_options = augmentation_config['options']
            self.aug_probs = augmentation_config['probs']
        print('Creating [{}] dataset:'.format(self.name))
        source_path = Path(os.path.join(cache_dir, source))
        if not source_path.exists():
            source_path.mkdir(parents=True, exist_ok=True)
        cache_path = Path(os.path.join(cache_dir, source, 'labels_cache'))
        if not source_path.exists():
            cache_path.mkdir(parents=True, exist_ok=True)
        config_path = f'**/data_configs/{source}/*.json'

        rows_to_remove_path = os.path.join(self.cache_dir, self.source, 'rows_to_remove.pickle')
        rows_to_remove = self.do_with_pickle_cache(lambda: [], rows_to_remove_path)

        self.files_with_labels = self.do_with_pickle_cache(
            lambda: self.get_files_with_labels(self.data_dir, config_path),
            os.path.join(cache_dir, source, 'files_with_labels.pickle'))
        self.remove_rows_from_files_with_labels(rows_to_remove)
        if self.size is None:
            self.size = len(self.files_with_labels)

        self.label_options_weights = self.do_with_pickle_cache(self.get_all_label_options_weights,
                                                               os.path.join(cache_dir, source,
                                                                            'label_options_weights.pickle'))
        base_prob = self.aug_probs['prob']
        sub_probs = self.aug_probs['sub_probs']
        for augmentation, augmentation_labels in self.aug_options.items():
            sub_prob = sub_probs[augmentation]['prob']
            option_prob = 1.0 / len(augmentation_labels)
            augmentation_true_options_weights = {'none': 0.0, 'disabled': (1 - base_prob) + base_prob * (1 - sub_prob),
                                                 **{
                                                     label: base_prob * sub_prob * option_prob for label in
                                                     augmentation_labels
                                                 }}
            augmentation_false_options_weights = {key: 1 - value for key, value in
                                                  augmentation_true_options_weights.items()}
            self.label_options_weights[augmentation] = {'true': augmentation_true_options_weights,
                                                        'false': augmentation_false_options_weights}

        all_label_groups = {key: [*label_groups[key], *augmentation_label_groups[key]] for key in label_groups.keys()}
        self.label_options_weights_groups = {
            key: {label: self.label_options_weights[label] for label in label_group}
            for key, label_group in all_label_groups.items()
        }

        self.label_options_groups = {
            key: {label: tuple(value['true'].keys()) for label, value in label_group.items()}
            for key, label_group in self.label_options_weights_groups.items()
        }

        self.label_options = {
            key: tuple(label_options['true'].keys())
            for key, label_options in self.label_options_weights.items()
        }

        self.label_weights_groups = {
            key: {label:
                      {'true': tuple(self.label_options_weights_groups[key][label]['true'][option] for option in options),
                       'false': tuple(self.label_options_weights_groups[key][label]['false'][option] for option in options)}
                  for label, options in label_group.items()}
            for key, label_group in self.label_options_groups.items()
        }

        self.label_weights = {
            label: {'true': tuple(self.label_options_weights[label]['true'][option] for option in options),
                  'false': tuple(self.label_options_weights[label]['false'][option] for option in options)}
            for label, options in self.label_options.items()
        }

        if self.should_augment:
            self.aug_methods = {
                'noise': NoiseAugmentor(self.aug_dir, self.label_options).augment,
                'rir': RirAugmentor(self.aug_dir).augment,
                'reverb': ReverbAugmentor(self.sampling_rate).augment,
                'lowpass': LowpassAugmentor(self.sampling_rate).augment,
                'highpass': HighpassAugmentor(self.sampling_rate).augment,
                'codec': CodecAugmentor(self.sampling_rate).augment,
                'hilbert': HilbertAugmentor(self.sampling_rate).augment
            }

        print('Dataset [{}] is ready!\n'.format(self.name))

    @staticmethod
    def do_with_pickle_cache(func, pickle_path):
        pickle_path = Path(pickle_path)
        if pickle_path.exists():
            with open(pickle_path, 'rb') as pickle_file:
                result = pickle.load(pickle_file)
        else:
            if not pickle_path.parent.exists():
                pickle_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                result = func()
                with open(pickle_path, 'wb') as pickle_file:
                    pickle.dump(result, pickle_file)
            except Exception as e:
                print(f'Failed to get item to pickle! [{func}], [{pickle_path}]')
                raise e
        return result

    @staticmethod
    def create_pickle_cache(func, pickle_path):
        pickle_path = Path(pickle_path)
        if not pickle_path.exists():
            if not pickle_path.parent.exists():
                pickle_path.parent.mkdir(parents=True, exist_ok=True)
            result = func()
            with open(pickle_path, 'wb') as pickle_file:
                pickle.dump(result, pickle_file)

    def get_all_label_options_weights(self):
        label_options = {}
        label_totals = {}

        with Pool(16) as pool:
            timed_labels_counts = pool.map(self.get_timed_labels_value_counts_by_index, range(len(self)))
        rows_to_remove_path = os.path.join(self.cache_dir, self.source, 'rows_to_remove.pickle')
        rows_to_remove = []
        valid_timed_labels_counts = timed_labels_counts.copy()
        for i, timed_labels_count in enumerate(timed_labels_counts):
            if isinstance(timed_labels_count, Exception):
                rows_to_remove.append(i)
                del valid_timed_labels_counts[i]
        self.create_pickle_cache(lambda: rows_to_remove, rows_to_remove_path)
        self.remove_rows_from_files_with_labels(rows_to_remove)

        for col in labels_to_use:
            col_value_counts = self.files_with_labels[col].value_counts()
            label_options[col] = {
                'true': dict(col_value_counts),
                'false': dict(-col_value_counts + len(self.files_with_labels))
            }
            label_totals[col] = len(self.files_with_labels)

        total_amount = len(valid_timed_labels_counts)
        for label in timed_labels_counts[0][0]:
            label_options[label] = {'true': {}, 'false': {}}
            label_totals[label] = total_amount
        for timed_labels_count in valid_timed_labels_counts:
            true_timed_labels_count, false_timed_labels_count = timed_labels_count
            for label in timed_labels_to_use:
                for key in true_timed_labels_count[label]:
                    true_value = true_timed_labels_count[label][key]
                    false_value = false_timed_labels_count[label][key]
                    if key not in label_options[label]['true']:
                        label_options[label]['true'][key] = 0
                        label_options[label]['false'][key] = total_amount
                    label_options[label]['true'][key] += 0 if true_value == 0 else 1
                    label_options[label]['false'][key] -= 1 if false_value == 0 else 0

        for label in label_options:
            total = label_totals[label]
            for key in label_options[label]['true']:
                label_options[label]['true'][key] /= total
                label_options[label]['false'][key] /= total
        label_options_weights = {key: {'true': self.sort_options(value['true'], none_ratio=0.0), 'false': self.sort_options(value['false'], none_ratio=1.0)} for key, value
                                 in label_options.items()}
        return label_options_weights

    def remove_rows_from_files_with_labels(self, rows_to_remove):
        if len(rows_to_remove) > 0:
            self.files_with_labels = self.files_with_labels.drop(rows_to_remove).reset_index(drop=True)

    def update_pickle_cache(self, func, path):
        path = Path(path)
        if not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
        result = func()
        with open(path, 'wb') as pickle_file:
            pickle.dump(result, pickle_file)

    def sort_options(self, options: dict, none_ratio=0.0):
        result = OrderedDict()
        if 'none' not in options:
            options['none'] = none_ratio
        ordered_keys = self.sort_options_keys(options)
        for key in ordered_keys:
            result[key] = options[key]
        return result

    def sort_options_keys(self, options: dict):
        items = list(options.items())
        items.sort(key=lambda pair: pair[1], reverse=True)
        keys = [key for key, value in items]
        if 'none' in keys:
            keys.remove('none')
        keys.insert(0, 'none')
        return keys

    def get_timed_labels_value_counts_by_index(self, i):
        try:
            currand = random.Random()
            if self.deterministic:
                currand.seed(i)
            if self.size < len(self.files_with_labels):
                i = (int(len(self.files_with_labels) / PHI) * i) % len(self.files_with_labels)
            labels, timed_labels = self.get_timed_labels(i)
            return self.get_labels_value_counts(timed_labels)
        except Exception as e:
            print('Item {} failed to get timed labels: [{}]'.format(i, e))
            return e

    def get_labels_value_counts(self, timed_labels):
        true_labels = {}
        false_labels = {}
        for label, timed_label in timed_labels.items():
            if label in timed_labels_to_use:
                timed_label['length'] = timed_label['end'] - timed_label['start']
                length_sum = timed_label['length'].sum()
                true_labels[label] = dict(timed_labels[label].groupby(['text'])['length'].sum())
                false_labels[label] = {key: length_sum - value for key, value in true_labels[label].items()}
        return true_labels, false_labels

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
                    complex_data = parse_complex_data(subdir, Path(self.data_dir), config_dict['config'],
                                                      config_dict['result'])
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
        timed_labels = parse_textgrid(self.data_dir, all_labels['textgrid'])
        timed_labels = self.add_extra_timed_labels(labels, timed_labels)
        timed_labels = {key: value for key, value in timed_labels.items() if key in timed_labels_to_use}
        return labels, timed_labels

    def add_extra_timed_labels(self, labels, timed_labels):
        timed_labels = self.add_sad_timed_labels(timed_labels)
        timed_labels = self.add_sad_based_timed_labels(labels, timed_labels)
        return timed_labels

    def add_sad_timed_labels(self, timed_labels):
        sad: pd.DataFrame = timed_labels['words'].copy()
        sad['text'] = sad['text'].apply(lambda x: 'silence' if x == '' else 'speech')
        timed_labels['sad'] = sad
        return timed_labels

    def add_sad_based_timed_labels(self, labels, timed_labels):
        for label in sad_based_labels:
            value = labels[label]
            timed_label = timed_labels['sad'].copy()
            timed_label['text'] = timed_label['text'].apply(lambda x: 'silence' if x == 'silence' else value)
            timed_labels[label] = timed_label
        return timed_labels

    def get_labels(self, index):
        labels = self.files_with_labels[labels_to_use].iloc[[index]].squeeze()
        return labels

    def get_grouped_labels(self, index):
        labels = self.get_labels(index)
        grouped_labels = {group: labels.filter(group_labels).to_dict() for group, group_labels in label_groups.items()}
        return grouped_labels

    def __getitem__(self, index):
        currand = random.Random()
        if self.deterministic:
            currand.seed(index)
        if self.size < len(self.files_with_labels):
            index = (int(len(self.files_with_labels) / PHI) * index) % len(self.files_with_labels)

        item = self.get_augmented_item(index, currand)
        return item

    def get_augmented_item(self, index, currand):
        wav, wav_path, time_labels, grouped_labels = self.get_cut_item(index, currand)
        if self.should_augment:
            wav, time_labels, grouped_labels = self.augment_item(wav, time_labels, grouped_labels, currand)
        return wav, wav_path, time_labels, grouped_labels

    def create_pickle_label(self, index):
        return self.create_pickle_cache(
            lambda: self.get_fresh_label(index),
            os.path.join(self.cache_dir, self.source, 'labels_cache', '{}.pickle'.format(index))
        )

    def get_pickle_label(self, index):
        return self.do_with_pickle_cache(
            lambda: self.get_fresh_label(index),
            os.path.join(self.cache_dir, self.source, 'labels_cache', '{}.pickle'.format(index))
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
        separate_timed_labels = [
            self.get_segmented_timed_labels_for_single(label_name, timed_label)
            for label_name, timed_label in timed_labels.items()
        ]
        return pd.concat(
            separate_timed_labels,
            axis=1
        )

    def get_segmented_timed_labels_for_single(self, label_name, timed_label):
        time_interval = self.embedding_size / self.sampling_rate
        start_time = timed_label.iloc[[0]].squeeze()['start']
        end_time = timed_label.iloc[[-1]].squeeze()['end']
        total_time = end_time - start_time
        segmented_length = int(total_time // time_interval)

        result_df = pd.DataFrame([{label_name: 'none'}] * segmented_length)

        for index, row in timed_label.iterrows():
            row_start_segment = int(row['start'] // time_interval)
            row_end_segment = int(row['end'] // time_interval)
            result_df[label_name][row_start_segment:row_end_segment] = row['text']
        return result_df

    def add_segmented_labels(self, segmented_timed_labels, labels):
        for col in labels.axes[0]:
            if col not in segmented_timed_labels:
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

    def get_wav(self, index):
        wav_path = get_path_by_glob(self.data_dir, self.files_with_labels.iloc[[index]].squeeze()['wav'])
        if self.disable_wavs:
            return torch.zeros((self.segment_length,)), str(wav_path)
        audio, sampling_rate = load_wav(wav_path)

        if sampling_rate != self.sampling_rate:
            raise ValueError("{} SR doesn't match target {} SR".format(
                sampling_rate, self.sampling_rate))

        audio = torch.FloatTensor(audio)
        return audio.squeeze(0), str(wav_path)

    def get_cut_item(self, index, currand):
        wav, wav_path = self.get_wav(index)
        pickle_label_groups = self.get_pickle_label(index)
        length = wav.size(0)
        embedded_segment_length = self.segment_length // self.embedding_size
        embedded_length = min(length // self.embedding_size,
                              next(iter(next(iter(pickle_label_groups.values())).values())).size(0))
        trimed_length = embedded_length * self.embedding_size
        trimed_start = 0
        if len(wav) > trimed_length:
            wav = wav[trimed_start:trimed_start + trimed_length]
        length = wav.size(0)
        # print(length, self.segment_length, embedded_length, embedded_segment_length)

        if length >= self.segment_length:
            max_embedded_start = embedded_length - embedded_segment_length
            embedded_start = currand.randint(0, max_embedded_start)
            start = embedded_start * self.embedding_size
            # print('trim: ', start, embedded_start)
        else:
            embedded_padding = embedded_segment_length - embedded_length
            prefix_embedded_padding = currand.randint(0, embedded_padding)
            postfix_embedded_padding = embedded_padding - prefix_embedded_padding
            padding = embedded_padding * self.embedding_size
            prefix_padding = prefix_embedded_padding * self.embedding_size
            postfix_padding = postfix_embedded_padding * self.embedding_size

        for key, group in pickle_label_groups.items():
            for label, label_item in group.items():
                label_item = label_item[0:embedded_length]
                if length >= self.segment_length:
                    cut_label_item = label_item[embedded_start:embedded_start + embedded_segment_length]
                else:
                    cut_label_item = torch.nn.functional.pad(label_item,
                                                             (prefix_embedded_padding, postfix_embedded_padding),
                                                             'constant')
                group[label] = cut_label_item

        if length >= self.segment_length:
            wav = wav[start:start + self.segment_length]
        else:
            wav = torch.nn.functional.pad(wav, (prefix_padding, postfix_padding), 'constant')

        grouped_labels = self.get_grouped_labels(index)
        return wav, wav_path, pickle_label_groups, grouped_labels

    def augment_item(self, cut_wav, cut_label, grouped_labels, currand):
        options = self.aug_options
        probs = self.aug_probs
        methods = self.aug_methods
        (length,) = next(iter(next(iter(cut_label.values())).values())).size()
        augmented_wav = cut_wav
        augmented_label = pd.DataFrame(['none'] * length, columns=['none'])
        should_augment = probs['prob'] > currand.random()
        for augmentation in options.keys():
            augmented_wav, augmented_label, value = self.augment_item_with(augmented_wav, augmented_label, cut_label,
                                                                           methods, options,
                                                                           probs, augmentation, currand, should_augment)
            for section, current_label_groups in augmentation_label_groups.items():
                if augmentation in current_label_groups:
                    grouped_labels[section][augmentation] = value
        augmentation_tensors = self.convert_segmented_labels_to_tensor(augmented_label, augmentation_label_groups)
        for key in cut_label.keys():
            current_augmentation = augmentation_tensors[key]
            for label, value in current_augmentation.items():
                cut_label[key][label] = value
        return augmented_wav, cut_label, grouped_labels

    def augment_item_with(self, augmented_wav, augmented_label, cut_label, methods, options, probs, aug_type, currand,
                          should=True):
        value = 'disabled'
        probs = probs['sub_probs'][aug_type]
        values = options[aug_type]
        aug_method = methods[aug_type]
        if should and probs['prob'] > currand.random():
            value = currand.choice(values)
            augmented_label, augmented_wav, value = aug_method(
                currand,
                augmented_label,
                cut_label,
                augmented_wav,
                value,
                self.disable_wavs
            )
        augmented_label[aug_type] = value
        return augmented_wav, augmented_label, value
