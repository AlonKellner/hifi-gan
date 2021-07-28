import torch
import torch.nn as nn

from configurable_module import get_module_from_config

LRELU_SLOPE = 0.1


class Generator(torch.nn.Module):
    def __init__(self, encoder, decoder):
        super(Generator, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, wave):
        split_e = self.encoder(wave)
        wave = self.decoder(split_e)
        return wave


class Encoder(torch.nn.Module):
    def __init__(self, vo_encoder, splitters):
        super(Encoder, self).__init__()
        self.vo_encoder = vo_encoder
        self.splitters = splitters

    def forward(self, wave):
        e = self.vo_encoder(wave)
        split_e = [splitter(e) for splitter in self.splitters]
        return split_e


class Decoder(torch.nn.Module):
    def __init__(self, mergers, vo_decoder):
        super(Decoder, self).__init__()
        self.mergers = mergers
        self.vo_decoder = vo_decoder

    def forward(self, split_e):
        split_e = [merger(single_e) for merger, single_e in
                   zip(self.mergers, split_e)]
        e = torch.stack(split_e, dim=0).sum(dim=0)
        wave = self.vo_decoder(e)
        return wave
