import torch


class TagsWrapper(torch.nn.Module):
    def __init__(self, model, tags):
        super(TagsWrapper, self).__init__()
        self.model = model
        self.tags = tags

    def forward(self, *x):
        return self.model(*x)
