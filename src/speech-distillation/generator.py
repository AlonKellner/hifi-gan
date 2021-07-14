import torch
import torch.nn as nn

from configurable_model import ConfigurableModel

LRELU_SLOPE = 0.1


class Generator(torch.nn.Module):
    def __init__(self, generator_config):
        super(Generator, self).__init__()
        encoder_config, decoder_config = generator_config
        self.encoder = Encoder(encoder_config)
        self.decoder = Decoder(decoder_config)

    def forward(self, wave):
        split_e = self.encoder(wave)
        wave = self.decoder(split_e)
        return wave


class Encoder(torch.nn.Module):
    def __init__(self, encoder_config):
        super(Encoder, self).__init__()
        vo_encoder_config, splitters_configs = encoder_config
        self.vo_encoder = ConfigurableModel(vo_encoder_config)
        self.splitters = nn.ModuleList([
            ConfigurableModel(splitter_config) for splitter_config in splitters_configs
        ])

    def forward(self, wave):
        e = self.vo_encoder(wave)
        split_e = torch.stack([splitter(e) for splitter in self.splitters], dim=0)
        return split_e


class Decoder(torch.nn.Module):
    def __init__(self, decoder_config):
        super(Decoder, self).__init__()
        mergers_configs, vo_decoder_config = decoder_config
        self.mergers = nn.ModuleList([
            ConfigurableModel(merger_config) for merger_config in mergers_configs
        ])
        self.vo_decoder = ConfigurableModel(vo_decoder_config)

    def forward(self, split_e):
        split_e = [merger(single_e.squeeze(dim=0)) for merger, single_e in
                   zip(self.mergers, split_e.split(1, dim=0))]
        e = torch.stack(split_e, dim=0).sum(dim=0)
        wave = self.vo_decoder(e)
        return wave
