import random

import torch
import torch.utils.data

from src.meldataset import load_wav


class WaveDataset(torch.utils.data.Dataset):
    def __init__(self, training_files, segment_length, sampling_rate, split=True, n_cache_reuse=1,
                 fine_tuning=False, deterministic=False):
        self.audio_files = training_files
        self.segment_length = segment_length
        self.sampling_rate = sampling_rate
        self.split = split
        self.n_cache_reuse = n_cache_reuse
        self.fine_tuning = fine_tuning
        self._cache_ref_count = 0
        self.deterministic = deterministic

    def __getitem__(self, index):
        filename = self.audio_files[index]
        if self._cache_ref_count == 0:
            audio, sampling_rate = load_wav(filename)
            self.cached_wav = audio
            if sampling_rate != self.sampling_rate:
                raise ValueError("{} SR doesn't match target {} SR".format(
                    sampling_rate, self.sampling_rate))
            self._cache_ref_count = self.n_cache_reuse
        else:
            audio = self.cached_wav
            self._cache_ref_count -= 1

        if audio.size(1) >= self.segment_length:
            max_audio_start = audio.size(1) - self.segment_length
            audio_start = 0 if self.deterministic else random.randint(0, max_audio_start)
            audio = audio[:, audio_start:audio_start+self.segment_length]
        else:
            audio = torch.nn.functional.pad(audio, (0, self.segment_length - audio.size(1)), 'constant')

        return audio.squeeze(0), filename

    def __len__(self):
        return len(self.audio_files)
