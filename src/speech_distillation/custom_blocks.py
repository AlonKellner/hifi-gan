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


class ChunkBlock(torch.nn.Module):
    def __init__(self, split_count):
        super(ChunkBlock, self).__init__()
        self.split_count = split_count

    def forward(self, x: torch.Tensor):
        split_x = x.chunk(self.split_count, dim=1)
        return split_x


class SplitBlock(torch.nn.Module):
    def __init__(self, split_size):
        super(SplitBlock, self).__init__()
        self.split_size = split_size

    def forward(self, x: torch.Tensor):
        split_x = x.split(self.split_size, dim=1)
        return split_x


class SplitDictBlock(torch.nn.Module):
    def __init__(self, split_mapping):
        super(SplitDictBlock, self).__init__()
        self.split_mapping = split_mapping

    def forward(self, x: torch.Tensor):
        split_x = {
            key: value for key, value
            in zip(self.split_mapping.keys(), x.split(list(self.split_mapping.values()), dim=1))
        }
        return split_x


class MergeBlock(torch.nn.Module):
    def __init__(self):
        super(MergeBlock, self).__init__()

    def forward(self, split_x):
        x = torch.cat(split_x, dim=1)
        return x


class MergeDictBlock(torch.nn.Module):
    def __init__(self):
        super(MergeDictBlock, self).__init__()

    def forward(self, split_x):
        x = torch.cat(list(split_x.values()), dim=1)
        return x


class ListBlock(torch.nn.Module):
    def __init__(self, sub_models):
        super(ListBlock, self).__init__()
        self.sub_models = sub_models

    def forward(self, split_x):
        split_x = [sub_model(x) for sub_model, x in zip(self.sub_models, split_x)]
        return split_x


class DictBlock(torch.nn.Module):
    def __init__(self, sub_models):
        super(DictBlock, self).__init__()
        self.sub_models = sub_models

    def forward(self, split_x):
        split_x = {key: self.sub_models[key](x) for key, x in split_x.items()}
        return split_x


class RecursiveBlock(torch.nn.Module):
    def __init__(self, complex_models):
        super(RecursiveBlock, self).__init__()
        self.complex_models = complex_models

    def forward(self, complex_x):
        return self.apply_recursive(self.complex_models, complex_x)

    def apply_recursive(self, complex_models, complex_x):
        if isinstance(complex_x, dict):
            return {key: self.apply_recursive(complex_models[key], value) for key, value in complex_x.items()}
        elif isinstance(complex_x, list):
            return [self.apply_recursive(complex_model, value) for value, complex_model in zip(complex_x, complex_models)]
        elif isinstance(complex_x, tuple):
            return tuple(self.apply_recursive(complex_model, value) for value, complex_model in zip(complex_x, complex_models))
        else:
            return complex_models(complex_x)


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

    def forward(self, *params):
        x = self.model(*params)
        features = self.features
        self.features = []
        processed_features = [
            self.feature_models[index % len(self.feature_models)](feature) for index, feature in enumerate(features)
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
