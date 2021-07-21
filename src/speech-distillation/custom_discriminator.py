import torch
import torch.nn as nn
import torch.nn.functional as F

from configurable_module import get_module_from_config


class AllInOneDiscriminator(nn.Module):
    def __init__(self, blocks_config):
        super(AllInOneDiscriminator, self).__init__()
        self.blocks = nn.ModuleList([AllInOneBlock(block_config) for block_config in blocks_config])

    def forward(self, raw):
        x = raw
        for block in self.blocks:
            x = block(x, raw)
        return x


class AllInOneBlock(nn.Module):
    def __init__(self, block_config):
        super(AllInOneBlock, self).__init__()
        before_block_config, raw_blocks_configs, after_block_config = block_config
        self.before_block = get_module_from_config(before_block_config)
        self.raw_blocks = nn.ModuleList([get_module_from_config(raw_block_config) for raw_block_config in raw_blocks_configs])
        self.after_block = get_module_from_config(after_block_config)

    def forward(self, x, raw):
        x = self.before_block(x)
        processed_raw_results = [raw_block(raw) for raw_block in self.raw_blocks]
        concatted = torch.cat([x, *processed_raw_results], dim=1)
        x = self.after_block(concatted)
        return x
