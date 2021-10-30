import torch


def feature_loss(fmap_r, fmap_g):
    loss = 0
    for fr, fg in zip(fmap_r, fmap_g):
        loss += torch.mean(torch.abs(fr - fg))
    return loss


def zero_loss(x):
    return torch.mean(x**2)


def one_loss(x):
    return torch.mean((1-x)**2)

