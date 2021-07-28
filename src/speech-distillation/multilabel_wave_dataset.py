import random

import torch
import torch.utils.data
from librosa.util import normalize

from src.meldataset import load_wav

MAX_WAV_VALUE = 32768.0


class MultilabelWaveDataset(torch.utils.data.Dataset):
    def __init__(self, training_files, segment_size, sampling_rate, split=True, n_cache_reuse=1,
                 fine_tuning=False, deterministic=False):
        self.audio_files = training_files
        self.segment_size = segment_size
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

        return audio.squeeze(0), filename

    def __len__(self):
        return len(self.audio_files)
