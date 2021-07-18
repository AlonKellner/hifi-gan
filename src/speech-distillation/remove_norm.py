from torch import nn
from torch.nn.utils.weight_norm import WeightNorm


def remove_weight_norm(module, name: str = 'weight'):
    for k, hook in module._forward_pre_hooks.items():
        if isinstance(hook, WeightNorm) and hook.name == name:
            hook.remove(module)
            del module._forward_pre_hooks[k]


def remove_module_weight_norms(module):
    for sub_module in module.children():
        if isinstance(sub_module, nn.Module):
            remove_module_weight_norms(sub_module)
        else:
            remove_weight_norm(module)
