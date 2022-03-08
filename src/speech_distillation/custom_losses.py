import torch
from torch.nn import functional as F

EPSILON = 1e-08
cached_losses = {'seg_bce': {}, 'seg_bias_bce': {}}


def recursive_loss(loss_func, x, *args):
    if isinstance(x, dict):
        return sum(recursive_loss(loss_func, x[key], *[arg[key] for arg in args]) for key in x.keys())
    elif isinstance(x, (list, tuple)):
        return sum(recursive_loss(loss_func, x[index], *[arg[index] for arg in args]) for index in range(len(x)))
    else:
        return loss_func(x, *args)


def minus_mean_loss(*x):
    return +sum(torch.mean(_x) for _x in x)


def plus_mean_loss(*x):
    return -sum(torch.mean(_x) for _x in x)


def seg_bce_loss(x, target, ratios: (float,) = None, dim=1):
    weights_hash = hash((ratios['true'], ratios['false']))
    bces = cached_losses['seg_bce']
    if weights_hash not in bces:
        bces[weights_hash] = SegBCELoss(ratios=ratios)
    x_t = x.transpose(dim, -1)
    one_hot_target = F.one_hot(target, x.size(dim)).float()

    return bces[weights_hash](x_t, one_hot_target)


def seg_bias_bce_loss(x, target, truth, ratios: (float,) = None, dim=1):
    weights_hash = hash((ratios['true'], ratios['false']))
    bces = cached_losses['seg_bias_bce']
    if weights_hash not in bces:
        bces[weights_hash] = SegBiasBCELoss(ratios=ratios)
    x_t = x.transpose(dim, -1)
    target_t = target.transpose(dim, -1)
    one_hot_truth = F.one_hot(truth, x.size(dim)).float()

    return bces[weights_hash](x_t, target_t, one_hot_truth)


class SegBiasBCELoss(torch.nn.Module):
    def __init__(self, ratios):
        super(SegBiasBCELoss, self).__init__()
        weights_tensors = ratios_to_weights_tensors(ratios)
        self.true_weights, self.false_weights = weights_tensors['true'], weights_tensors['false']

    def forward(self, x, target, truth):
        high = torch.max(truth, target)
        low = torch.min(truth, target)
        scale = high - low + EPSILON
        x_norm = (x - low) / scale
        x_clamped = torch.clamp(x_norm, min=0, max=1)

        biased_cross_entropy = F.binary_cross_entropy(x_clamped, 1-truth, reduction='none') * (scale * scale)

        total_loss = normalize_segmentation_loss(biased_cross_entropy, target, self.true_weights, self.false_weights)
        return total_loss


def ratios_to_weights_tensors(ratios):
    return {key: ratios_to_weights_tensor(value) for key, value in ratios.items()}

def ratios_to_weights_tensor(ratios):
    smallest_not_0 = min([r for r in ratios if r != 0], default=EPSILON)
    ratios_t = torch.Tensor(ratios)
    ratios_t = torch.where(ratios_t == 0, torch.ones_like(ratios_t)*smallest_not_0, ratios_t)
    weights = ratios_t**-1
    return weights.cuda()


class SegBCELoss(torch.nn.Module):
    def __init__(self, ratios, batch_dim=0, class_dim=2):
        super(SegBCELoss, self).__init__()
        self.batch_dim = batch_dim
        self.class_dim = class_dim
        weights_tensors = ratios_to_weights_tensors(ratios)
        self.true_weights, self.false_weights = weights_tensors['true'], weights_tensors['false']

    def forward(self, x, target):
        cross_entropy = F.binary_cross_entropy(x, target, reduction='none')
        total_loss = normalize_segmentation_loss(cross_entropy, target, self.true_weights, self.false_weights)
        return total_loss


class SimpleCosineLoss(torch.nn.Module):
    def __init__(self):
        super(SimpleCosineLoss, self).__init__()
        self.cos = torch.nn.CosineEmbeddingLoss()

    def forward(self, x, target):
        flat_x = torch.flatten(x, start_dim=1)
        flat_target = torch.flatten(target, start_dim=1)
        return self.cos(flat_x, flat_target, torch.ones(flat_x.size(0), device=flat_x.device))


def normalize_segmentation_loss(loss, target, true_weights, false_weights, batch_dim=0, class_dim=2):
    sum_dims = [i for i in range(len(target.size())) if i != class_dim and i != batch_dim]

    true_target = target
    false_target = 1 - target

    true_per_class_loss = (true_target * loss).sum(dim=sum_dims) / (true_target.sum(dim=sum_dims) + 1)
    false_per_class_loss = (false_target * loss).sum(dim=sum_dims) / (false_target.sum(dim=sum_dims) + 1)

    weighted_true_per_class_loss = true_per_class_loss * true_weights
    weighted_false_per_class_loss = false_per_class_loss * false_weights

    total_loss = weighted_true_per_class_loss.mean() + weighted_false_per_class_loss.mean()
    return total_loss


loss_types = {
    '-': lambda: minus_mean_loss,
    '+': lambda: plus_mean_loss,
    'seg_bce': lambda: seg_bce_loss,
    'seg_bias_bce': lambda: seg_bias_bce_loss,
    'l2': torch.nn.MSELoss,
    'l1': torch.nn.L1Loss,
    'cos': SimpleCosineLoss
}


def get_loss_by_type(loss_type):
    return loss_types[loss_type]()


def get_losses_by_types(loss_type):
    if isinstance(loss_type, dict):
        return {key: get_losses_by_types(current_type) for key, current_type in loss_type.items()}
    else:
        return get_loss_by_type(loss_type)
