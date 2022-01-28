from src.speech_distillation.configurable_module import get_module_from_config
from src.speech_distillation.static_configs import get_classifier_backbone
import torch


def generate_keepers_by_example(input_channels, example_item, cache_hook=lambda k, x: x(), layers=[]):
    keepers_configs = generate_keepers_config_by_example(input_channels, example_item, layers=layers)
    return torch.nn.ModuleDict(
        {key: get_module_from_config(cache_hook(key, lambda: keeper_config)) for key, keeper_config in
         keepers_configs.items()}
    )


def generate_keepers_config_by_example(input_channels, grouped_examples, layers=[], ensemble_size=3):
    keepers = {
        key: ('ensemble', [
            generate_classifier_by_example(
                input_channels,
                {key2: value2 for key2, value2 in grouped_examples.items() if key == key2},
                layers=layers
            ) for i in range(ensemble_size)
        ])
        for key, example_item in grouped_examples.items()
    }
    return keepers


def generate_hunters_by_example(input_channels, example_item, cache_hook=lambda k, x: x(), layers=[]):
    hunters_configs = generate_hunters_config_by_example(input_channels, example_item, layers=layers)
    return torch.nn.ModuleDict(
        {key: get_module_from_config(cache_hook(key, lambda: hunter_config)) for key, hunter_config in
         hunters_configs.items()}
    )


def generate_hunters_config_by_example(input_channels, grouped_examples, layers, ensemble_size=3):
    hunters = {
        key: ('ensemble', [
            generate_classifier_by_example(
                input_channels,
                {key2: value2 for key2, value2 in grouped_examples.items() if key != key2},
                layers=layers
            ) for i in range(ensemble_size)
        ])
        for key, example_item in grouped_examples.items()
    }
    return hunters


def generate_classifier_by_example(input_channels, example, layers):
    label_groups = {
        ex_key: {key: len(value) for key, value in label_group.items()}
        for ex_key, label_group in example.items()
    }
    groups_channels = {
        ex_key: sum(value for value in label_group.values())
        for ex_key, label_group in label_groups.items()
    }
    output_channels = sum(groups_channels.values())
    return [
        get_classifier_backbone(input_channels, output_channels, layers=layers),
        ('split', groups_channels),
        ('recursive', {group: ('split', sizes) for group, sizes in label_groups.items()}),
        ('recursive', {group: {key: ('softmax',) for key in sizes} for group, sizes in label_groups.items()}),
    ]
