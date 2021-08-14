import torch
import torch.nn as nn

LRELU_SLOPE = 0.1


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
