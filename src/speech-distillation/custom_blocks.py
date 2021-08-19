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


class SumBlock(torch.nn.Module):
    def __init__(self, sub_blocks):
        super(SumBlock, self).__init__()
        self.sub_blocks = sub_blocks

    def forward(self, x):
        xsum = None
        for sub_block in self.sub_blocks:
            value = sub_block(x)
            if xsum is None:
                xsum = value
            else:
                xsum += value
        x = xsum
        return x


class ResBlock(torch.nn.Module):
    def __init__(self, model):
        super(ResBlock, self).__init__()
        self.model = model

    def forward(self, x):
        delta = self.model(x)
        x = x + delta
        return x


class SplitterBlock(torch.nn.Module):
    def __init__(self, splitters):
        super(SplitterBlock, self).__init__()
        self.splitters = splitters

    def forward(self, x):
        split_x = [splitter(x) for splitter in self.splitters]
        return split_x


class MergerBlock(torch.nn.Module):
    def __init__(self, mergers):
        super(MergerBlock, self).__init__()
        self.mergers = mergers

    def forward(self, split_x):
        split_x = [merger(single_x) for merger, single_x in
                   zip(self.mergers, split_x)]
        x = torch.stack(split_x, dim=0).sum(dim=0)
        return x


class ValveBlock(torch.nn.Module):
    def __init__(self, ratio=1):
        super(ValveBlock, self).__init__()
        self.ratio = ratio

    def forward(self, x):
        return x * self.ratio


class SubResBlock(torch.nn.Module):
    def __init__(self, model):
        super(SubResBlock, self).__init__()
        self.model = model

    def forward(self, x):
        delta = self.model(x)
        x = x - delta
        return x


class FeatureBlock(torch.nn.Module):
    def __init__(self, model, tags_to_find):
        super(FeatureBlock, self).__init__()
        self.model = model
        self.tags_to_find = tags_to_find
        self.hooks = []
        self.model.apply(self._register_hook)
        self.features = []

    def _register_hook(self, module):
        def hook(hooked_module, module_input, output):
            self.features.append(output)
        if hasattr(module, 'tags'):
            if any(tag in self.tags_to_find for tag in module.tags):
                self.hooks.append(module.register_forward_hook(hook))

    def forward(self, *x):
        x = self.model(*x)
        features = self.features
        self.features = []
        return x, features


class ProcessedFeatureBlock(FeatureBlock):
    def __init__(self, model, tags_to_find, feature_models):
        super(ProcessedFeatureBlock, self).__init__(model, tags_to_find)
        self.feature_models = feature_models

    def forward(self, *x):
        x = self.model(*x)
        features = self.features
        self.features = []
        processed_features = [
            feature_model(feature) for feature_model, feature in zip(self.feature_models, features)
        ]
        return x, processed_features


def get_modules(model, module_type, tags_to_find=None):
    return list(filter(lambda module: is_module_valid(module, module_type, tags_to_find), model.modules()))


def is_module_valid(module, module_type, tags_to_find=None):
    if not isinstance(module, module_type):
        return False
    if tags_to_find is None:
        return True
    if not hasattr(module, 'tags'):
        return False
    tags_were_found = any(tag in tags_to_find for tag in module.tags)
    return tags_were_found
