import torch


def expand(tensor, size, dim):
    size_left = size
    unrolled_tensors = []
    while size_left > tensor.size(dim):
        unrolled_tensors.append(tensor)
        size_left -= tensor.size(dim)
    unrolled_tensors.append(torch.narrow(tensor, dim=dim, start=0, length=size_left))
    cat_tensor = torch.cat(unrolled_tensors, dim=dim)
    return cat_tensor


def mix(tensor, rolls, dim):
    narrowed_tensors = torch.split(tensor, rolls, dim=dim)
    rolled_tensors = [torch.roll(narrowed_tensor, roll, dims=dim) for roll, narrowed_tensor in enumerate(narrowed_tensors)]
    cat_tensor = torch.cat(rolled_tensors, dim=dim)
    return cat_tensor


def unmix(tensor, rolls, dim):
    narrowed_tensors = torch.split(tensor, rolls, dim=dim)
    rolled_tensors = [torch.roll(narrowed_tensor, -roll, dims=dim) for roll, narrowed_tensor in enumerate(narrowed_tensors)]
    cat_tensor = torch.cat(rolled_tensors, dim=dim)
    return cat_tensor
