from src.speech_distillation.configurable_module import get_module_from_config
from src.speech_distillation.static_configs import get_classifier_backbone
import torch


def generate_keepers_by_example(input_channels, example_item):
    keepers_configs = generate_keepers_config_by_example(input_channels, example_item)
    return torch.nn.ModuleDict(
        {key: get_module_from_config(keeper_config) for key, keeper_config in keepers_configs.items()}
    )


def generate_keepers_config_by_example(input_channels, grouped_examples):
    keepers = {
        key: generate_keeper_config_by_example(input_channels, example_item)
        for key, example_item in grouped_examples.items()
    }
    return keepers


def generate_keeper_config_by_example(input_channels, example, hiddens=[1092, 546, 364]):
    output_channels = sum(example.values())
    return [
        get_classifier_backbone(input_channels, output_channels, hiddens),
        ('split', example)
    ]


def generate_hunters_by_example(input_channels, example_item):
    hunters_configs = generate_hunters_config_by_example(input_channels, example_item)
    return torch.nn.ModuleDict(
        {key: get_module_from_config(hunter_config) for key, hunter_config in hunters_configs.items()})


def generate_hunters_config_by_example(input_channels, grouped_examples):
    hunters = {
        key: generate_hunter_by_example(
            input_channels,
            {key2: value2 for key2, value2 in grouped_examples.items() if key != key2}
        )
        for key, example_item in grouped_examples.items()
    }
    return hunters


def generate_hunter_by_example(input_channels, example, hiddens=[1092, 546, 364]):
    label_groups = {ex_key: value for ex_key, value in example.items()}
    groups_channels = {
        ex_key: sum(value for value in label_group.values())
        for ex_key, label_group in label_groups.items()
    }
    output_channels = sum(groups_channels.values())
    return [
        get_classifier_backbone(input_channels, output_channels, hiddens),
        ('split', groups_channels),
        ('recursive', {group: ('split', sizes) for group, sizes in label_groups.items()})
    ]
