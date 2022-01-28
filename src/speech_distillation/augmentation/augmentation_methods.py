from pathlib import Path

import torch
from torch.nn import functional as F
from torchaudio import functional as AF, sox_effects
import torchaudio

from src.meldataset import load_wav

torchaudio.set_audio_backend('sox_io')


def apply_sox_effect(tensor, sample_rate, effects):
    return sox_effects.apply_effects_tensor(tensor=tensor, sample_rate=sample_rate, effects=effects)


class NoiseAugmentor:
    def __init__(self, dir, label_options, min_nsr=0.05, max_nsr=0.2):
        self.max_nsr = max_nsr
        self.min_nsr = min_nsr
        self.nsr_range = max_nsr - min_nsr
        mic_brands = Path(dir).glob('libriadapt/noise/*')
        self.noise_paths = {
            mic_brand.stem: {
                noise_type.stem: list(noise_type.glob('**/*.wav')) for noise_type in mic_brand.glob('*')
            } for mic_brand in mic_brands
        }
        self.label_options = label_options

    def augment(self, random, labels, cut_labels, wav, noise_type, disable_wav_augmentation=False):
        mic_brand = self.label_options['mic-brand'][cut_labels['style']['mic-brand'][0].item()]
        if mic_brand not in self.noise_paths or noise_type not in self.noise_paths[mic_brand]:
            noise_type = 'disabled'
        else:
            if not disable_wav_augmentation:
                noise_options = self.noise_paths[mic_brand][noise_type]
                noise_path = random.choice(noise_options)
                noise, sampling_rate = load_wav(noise_path)
                nsr = (self.min_nsr + random.random() * self.nsr_range)
                snr = 1 - nsr
                noise = noise * nsr
                wav = wav * snr
                if noise.size(0) > wav.size(0):
                    noise = noise[0][0:wav.size(0)]
                else:
                    noise = F.pad(noise[None, ...], (0, wav.size(0) - noise.size(1)), 'circular')[0][0]
                wav = wav + noise
        return labels, wav, noise_type


class RirAugmentor:
    def __init__(self, dir):
        rir_options = Path(dir).glob('RIRS_NOISES/simulated_rirs/*')
        self.rir_paths = {
            rir_option.stem: list(rir_option.glob('**/*.wav')) for rir_option in rir_options
        }

    def augment(self, random, labels, cut_labels, wav, aug_parameter, disable_wav_augmentation=False):
        if not disable_wav_augmentation:
            rir_options = self.rir_paths[aug_parameter]
            rir_path = random.choice(rir_options)
            rir, sampling_rate = load_wav(rir_path)
            rir = rir / torch.norm(rir, p=2)
            rir = torch.flip(rir, [1])
            max_sample_index = rir.argmax(dim=1)[0].item()
            rir = rir[0:max_sample_index]
            wav = F.pad(wav, (rir.shape[1] - 1, 0))
            wav = F.conv1d(wav[None, None, ...], rir[None, ...])[0][0]
        return labels, wav, aug_parameter


class CodecAugmentor:
    def __init__(self, sample_rate):
        self.codecs_parameters = {
            'wav': {'encoding': 'ULAW', "bits_per_sample": 8},
            'gsm': {},
            'mp3': {'compression': -9},
            'vorbis': {'compression': -1},
        }
        self.sample_rate = sample_rate

    def augment(self, random, labels, cut_labels, wav, aug_parameter, disable_wav_augmentation=False):
        if not disable_wav_augmentation:
            original_length = wav.size(0)
            wav = wav[None, ...]
            sample_rate = 8000 if aug_parameter == 'gsm' else self.sample_rate
            wav = AF.apply_codec(
                wav,
                sample_rate=sample_rate,
                format=aug_parameter,
                **self.codecs_parameters[aug_parameter]
            )[0]
            wav = wav[0:original_length]
        return labels, wav, aug_parameter


class LowpassAugmentor:
    def __init__(self, sample_rate):
        self.sample_rate = sample_rate

    def augment(self, random, labels, cut_labels, wav, aug_parameter, disable_wav_augmentation=False):
        if not disable_wav_augmentation:
            wav = wav.unsqueeze(0)
            wav, sr = apply_sox_effect(wav, self.sample_rate, [['lowpass', '-1', aug_parameter]])
            wav = wav.squeeze(0)
        return labels, wav, aug_parameter


class HighpassAugmentor:
    def __init__(self, sample_rate):
        self.sample_rate = sample_rate

    def augment(self, random, labels, cut_labels, wav, aug_parameter, disable_wav_augmentation=False):
        if not disable_wav_augmentation:
            wav = wav.unsqueeze(0)
            wav, sr = apply_sox_effect(wav, self.sample_rate, [['highpass', '-1', aug_parameter]])
            wav = wav.squeeze(0)
        return labels, wav, aug_parameter


class ReverbAugmentor:
    def __init__(self, sample_rate):
        self.sample_rate = sample_rate

    def augment(self, random, labels, cut_labels, wav, aug_parameter, disable_wav_augmentation=False):
        if not disable_wav_augmentation:
            wav = wav.unsqueeze(0)
            wav, sr = apply_sox_effect(wav, self.sample_rate, [['reverb', '-w']])
            wav = wav.squeeze(0)
            wav = wav[0]
        return labels, wav, aug_parameter


class HilbertAugmentor:
    def __init__(self, sample_rate):
        self.sample_rate = sample_rate

    def augment(self, random, labels, cut_labels, wav, aug_parameter, disable_wav_augmentation=False):
        if not disable_wav_augmentation:
            wav = wav.unsqueeze(0)
            wav, sr = apply_sox_effect(wav, self.sample_rate, [['hilbert']])
            wav = wav.squeeze(0)
        return labels, wav, aug_parameter
