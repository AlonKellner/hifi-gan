from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import glob
import json
import os

import torch
from scipy.io.wavfile import write

from remove_norm import remove_module_weight_norms
from generator import Generator
from src.env import AttrDict
from src.meldataset import MAX_WAV_VALUE, load_wav
from static_configs import get_static_generator_config

device = None


def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict


def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '*')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return ''
    return sorted(cp_list)[-1]


def inference(a, h):
    generator = Generator(get_static_generator_config()).to(device)

    state_dict_g = load_checkpoint(a.checkpoint_file, device)
    generator.load_state_dict(state_dict_g['generator'])

    filelist = os.listdir(a.input_wavs_dir)

    os.makedirs(a.output_dir, exist_ok=True)

    generator.eval()
    remove_module_weight_norms(generator)
    with torch.no_grad():
        for i, filename in enumerate(filelist):
            wav, sr = load_wav(os.path.join(a.input_wavs_dir, filename))
            wav = wav / MAX_WAV_VALUE
            wav = torch.FloatTensor(wav).to(device)
            y_g_hat = generator(wav.unsqueeze(0).unsqueeze(0))
            audio = y_g_hat.squeeze()
            audio = audio * MAX_WAV_VALUE
            audio = audio.cpu().numpy().astype('int16')

            output_file = os.path.join(a.output_dir, os.path.splitext(filename)[0] + '_generated.wav')
            write(output_file, h.sampling_rate, audio)
            print(output_file)


def main():
    print('Initializing Inference Process..')

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_wavs_dir', default='test_files')
    parser.add_argument('--output_dir', default='generated_files')
    parser.add_argument('--checkpoint_file', required=True)
    a = parser.parse_args()

    config_file = os.path.join(os.path.split(a.checkpoint_file)[0], 'config.json')
    with open(config_file) as f:
        data = f.read()

    json_config = json.loads(data)
    config = AttrDict(json_config)

    torch.manual_seed(config.seed)
    global device
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    inference(a, config)


if __name__ == '__main__':
    main()

