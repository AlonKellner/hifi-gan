import random

import torch
import torch.utils.data
from librosa.util import normalize
from pathlib import Path
import re
import json
from csv import reader

from src.meldataset import load_wav

MAX_WAV_VALUE = 32768.0


class MultilabelWaveDataset(torch.utils.data.Dataset):
    def __init__(self, dir, config_path, segment_size, sampling_rate, split=True, n_cache_reuse=1,
                 fine_tuning=False, deterministic=False):
        self.segment_size = segment_size
        self.sampling_rate = sampling_rate
        self.split = split
        self.n_cache_reuse = n_cache_reuse
        self.fine_tuning = fine_tuning
        self._cache_ref_count = 0
        self.deterministic = deterministic
        self.files_with_labels = self.get_files_with_labels(dir, config_path)

    def get_files_with_labels(self, dir, config_path):
        subdir_list = [path for path in Path(dir).glob('*/')]
        results = []
        for subdir in subdir_list:
            config_list = [path for path in subdir.glob(config_path)]
            if len(config_list) == 0:
                raise Exception('Missing config [{}] in [{}]'.format(config_path, str(subdir)))
            print('Config [{}] found! Using dir [{}]'.format(config_path, str(subdir)))
            config_file = min(config_list, key=lambda x: len(str(x)))
            config = config_file.read_text()
            config_dict = json.loads(config)
            wav_list_source = config_dict['source']
            ignored_files = 0
            wav_list = self._get_paths_from_source(subdir, wav_list_source)
            if len(wav_list) == 0:
                raise Exception('Missing wavs in [{}]'.format(str(subdir)))
            for wav in wav_list:
                wav_path = str(wav)
                file_name_match = re.search(filename_pattern, str(wav))
                if file_name_match is None:
                    print('Label pattern [{}] invalid with [{}], ignoring file'.format(filename_pattern, wav.stem))
                else:
                    resolved_labels = {label: file_name_match.group(label) for label in labels}
                    results.append((wav_path, resolved_labels))
            if ignored_files > 0:
                print('[{}] files ignored in [{}]'.format(ignored_files, str(subdir)))
        return results

    def __getitem__(self, index):
        filename, labels = self.files_with_labels[index]
        if self._cache_ref_count == 0:
            audio, sampling_rate = load_wav(filename)
            audio = audio / MAX_WAV_VALUE
            if not self.fine_tuning:
                audio = normalize(audio) * 0.95
            self.cached_wav = audio
            if sampling_rate != self.sampling_rate:
                raise ValueError("{} SR doesn't match target {} SR".format(
                    sampling_rate, self.sampling_rate))
            self._cache_ref_count = self.n_cache_reuse
        else:
            audio = self.cached_wav
            self._cache_ref_count -= 1

        audio = torch.FloatTensor(audio)
        audio = audio.unsqueeze(0)

        if audio.size(1) >= self.segment_size:
            max_audio_start = audio.size(1) - self.segment_size
            audio_start = 0 if self.deterministic else random.randint(0, max_audio_start)
            audio = audio[:, audio_start:audio_start+self.segment_size]
        else:
            audio = torch.nn.functional.pad(audio, (0, self.segment_size - audio.size(1)), 'constant')

        return audio.squeeze(0), labels, filename

    def _get_paths_from_source(self, subdir, source):
        if source['type'] == 'glob':
            return self._get_paths_from_glob(subdir, source['path'])
        elif source['type'] == 'csv':
            return self._get_paths_from_csv(subdir, source['path'])

    @staticmethod
    def _get_paths_from_glob(subdir, path):
        return list(subdir.glob(path))

    @staticmethod
    def _get_paths_from_csv(subdir, path):
        file_path = min(subdir.glob(path), key=lambda x: len(str(x)))
        with open(file_path, 'r') as file_stream:
            csv_reader = reader(file_stream)
            return [row[0] for row in csv_reader]

    def __len__(self):
        return len(self.files_with_labels)
