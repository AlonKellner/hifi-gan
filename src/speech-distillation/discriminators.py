import torch
import torch.nn as nn
import torch.nn.functional as F

from configurable_module import get_module_from_config

LRELU_SLOPE = 0.1


class DiscriminatorP(torch.nn.Module):
    def __init__(self, period, layers_config):
        super(DiscriminatorP, self).__init__()
        self.period = period
        self.fmaps = [
            should_map for _, should_map in layers_config
        ]
        self.layers = nn.ModuleList([get_module_from_config(layer_config) for layer_config, _ in layers_config])

    def forward(self, x):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0: # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for i, item in enumerate(zip(self.layers, self.fmaps)):
            layer, should_fmap = item
            x = layer(x)
            if should_fmap:
                fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self, discriminator_configs):
        super(MultiPeriodDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList(list(map(DiscriminatorP, discriminator_configs)))

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class DiscriminatorS(torch.nn.Module):
    def __init__(self, layers_config):
        super(DiscriminatorS, self).__init__()
        self.fmaps = [
            should_map for _, should_map in layers_config
        ]
        self.layers = nn.ModuleList([get_module_from_config(layer_config) for layer_config, _ in layers_config])

    def forward(self, x):
        fmap = []
        for i, item in enumerate(zip(self.layers, self.fmaps)):
            layer, should_fmap = item
            x = layer(x)
            if should_fmap:
                fmap.append(x)
        x = torch.flatten(x, 1, -1)
        return x, fmap


class MultiScaleDiscriminator(torch.nn.Module):
    def __init__(self, discriminator_configs):
        super(MultiScaleDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList(list(map(DiscriminatorS, discriminator_configs)))

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, discriminator in enumerate(self.discriminators):
            y_d_r, fmap_r = discriminator(y)
            y_d_g, fmap_g = discriminator(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs
