import torch
import torch.nn as nn
import torch.nn.functional as f


class AllInOneDiscriminator(nn.Module):
    def __init__(self, pre_model, blocks, post_model):
        super(AllInOneDiscriminator, self).__init__()
        self.pre_model = pre_model
        self.blocks = blocks
        self.post_model = post_model

    def forward(self, raw):
        x = self.pre_model(raw)
        for block in self.blocks:
            x = block(x, raw)
        return self.post_model(x)


class AllInOneBlock(nn.Module):
    def __init__(self, before_block, raw_blocks, after_block, padding_mode='constant', padding_value=0):
        super(AllInOneBlock, self).__init__()
        self.before_block = before_block
        self.raw_blocks = raw_blocks if raw_blocks is not None else []
        self.after_block = after_block
        self.padding_mode = padding_mode
        self.padding_value = padding_value

    def forward(self, x, raw):
        x = self.before_block(x)
        processed_raw_results = [raw_block(raw) for raw_block in self.raw_blocks]
        all_results = [x, *processed_raw_results]
        lengths = [result.size()[2] for result in all_results]
        max_length = max(lengths)
        paddings = [max_length - length for length in lengths]
        padded_results = [f.pad(result, (0, padding), self.padding_mode, self.padding_value) for
                          result, padding in zip(all_results, paddings)]
        concatted = torch.cat(padded_results, dim=1)
        x = self.after_block(concatted)
        return x
