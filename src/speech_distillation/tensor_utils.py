import torch


def expand_unroll(tensor, size, dim):
    size_left = size
    unrolled_tensors = []
    while size_left > tensor.size(dim):
        unrolled_tensors.append(tensor)
        size_left -= tensor.size(dim)
    unrolled_tensors.append(torch.narrow(tensor, dim=dim, start=0, length=size_left))
    cat_tensor = torch.cat(unrolled_tensors, dim=dim)
    return cat_tensor


def cycles_roll(tensor, rolls, dim):
    size = tensor.size()
    double_tensor = torch.cat([tensor, tensor], dim=dim)
    last_rolled_index = 0
    rolled_tensors = []
    for roll, length in enumerate(rolls):
        narrowed_tensor = torch.narrow(double_tensor, dim=dim, start=last_rolled_index, length=length)
        rolled_tensor = torch.roll(narrowed_tensor, roll, dims=dim)
        rolled_tensors.append(rolled_tensor)
        last_rolled_index += length
        last_rolled_index = last_rolled_index % size[dim]
    cat_tensor = torch.cat(rolled_tensors, dim=dim)
    return cat_tensor
