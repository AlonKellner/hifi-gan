import math

import torch
from torch.nn import functional as F


def recursive_loss(loss_func, x, *args):
    if isinstance(x, dict):
        return sum(recursive_loss(loss_func, x[key], *[arg[key] for arg in args]) for key in x.keys()) / len(x)
    elif isinstance(x, (list, tuple)):
        return sum(recursive_loss(loss_func, x[index], *[arg[index] for arg in args]) for index in range(len(x))) / len(
            x)
    else:
        return loss_func(x, *args)


def minus_mean_loss(x):
    return torch.mean(x)


def plus_mean_loss(x):
    return -torch.mean(x)


def cross_entropy_loss_func():
    return torch.nn.CrossEntropyLoss()


def binary_cross_entropy_loss(x, y):
    return -torch.mean(binary_entropy(x, y, 1) + binary_entropy(x, y, -1))


def binary_entropy(x, y, sign):
    return (y * 0.5 * sign + 0.5) * torch.log(x * 0.5 * sign + 0.5 + 1e-08)


def bias_corrected_cross_entropy_loss(x, target, ground_truth, dim=1):
    flat = bias_corrected_cross_entropy(x, target, ground_truth, dim=dim)
    loss = flat.mean()
    return loss


def bias_corrected_cross_entropy(x, target, ground_truth, dim=1):
    one_hot = F.one_hot(ground_truth, x.size(dim)).transpose(-1, dim)
    high = torch.max(one_hot, target)
    low = torch.min(one_hot, target)
    scale = high - low
    normalized = (x - low) / scale
    sign = (-one_hot * 2 + 1)
    transformed = sign * (normalized - 0.5) + 0.5
    raw = -torch.log(transformed + 1e-08)
    flat = torch.max(raw, torch.zeros_like(raw))
    flat_scaled = flat * scale * scale
    return flat_scaled


loss_types = {
    '-': lambda: minus_mean_loss,
    'minus': lambda: minus_mean_loss,
    '+': lambda: plus_mean_loss,
    'plus': lambda: plus_mean_loss,
    'ce': torch.nn.CrossEntropyLoss,
    'cross_entropy': torch.nn.CrossEntropyLoss,
    'bce': lambda: binary_cross_entropy_loss,
    'binary_cross_entropy': lambda: binary_cross_entropy_loss,
    'mse': torch.nn.MSELoss,
    'mae': torch.nn.L1Loss,
    'l2': torch.nn.MSELoss,
    'l1': torch.nn.L1Loss,
    'bias_ce': lambda: bias_corrected_cross_entropy_loss,
    'bias_cross_entropy': lambda: bias_corrected_cross_entropy_loss,
}


def get_loss_by_type(loss_type):
    return loss_types[loss_type]()


def get_losses_by_types(loss_type):
    if isinstance(loss_type, dict):
        return {key: get_losses_by_types(current_type) for key, current_type in loss_type.items()}
    else:
        return get_loss_by_type(loss_type)
