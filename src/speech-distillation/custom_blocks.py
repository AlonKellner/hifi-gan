import torch


class FusionBlock(torch.nn.Module):
    def __init__(self, sub_blocks):
        super(FusionBlock, self).__init__()
        self.sub_blocks = sub_blocks

    def forward(self, x):
        xsum = None
        for sub_block in self.sub_blocks:
            value = sub_block(x)
            if xsum is None:
                xsum = value
            else:
                xsum += value
        x = xsum / len(self.sub_blocks)
        return x


class ResBlock(torch.nn.Module):
    def __init__(self, model):
        super(ResBlock, self).__init__()
        self.model = model

    def forward(self, x):
        delta = self.model(x)
        x = x + delta
        return x
