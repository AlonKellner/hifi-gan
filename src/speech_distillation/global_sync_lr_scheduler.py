import math

import torch

EPOCH_DEPRECATION_WARNING = (
    "The epoch parameter in `scheduler.step()` was not necessary and is being "
    "deprecated where possible. Please use `scheduler.step()` to step the "
    "scheduler. During the deprecation, if epoch is different from None, the "
    "closed form is used instead of the new chainable form, where available. "
    "Please open an issue if you are unable to replicate your use case: "
    "https://github.com/pytorch/pytorch/issues/new/choose."
)


class LRScheduler(torch.optim.lr_scheduler._LRScheduler):

    @property
    def optimizer(self):
        raise NotImplementedError

    def state_dict(self):
        raise NotImplementedError

    def load_state_dict(self, state_dict):
        raise NotImplementedError

    def get_last_lr(self):
        raise NotImplementedError

    def get_lr(self):
        raise NotImplementedError

    def print_lr(self, is_verbose, group, lr, epoch=None):
        raise NotImplementedError

    def step(self, epoch=None):
        raise NotImplementedError


class GlobalSyncDecoratorLR(LRScheduler):
    def __init__(self, pl_module, inner_lr):
        self.pl_module = pl_module
        self.inner_lr = inner_lr
        self.steps = 0

    @property
    def optimizer(self):
        return self.inner_lr.optimizer

    def state_dict(self, *args, **kwargs):
        return self.inner_lr.state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        return self.inner_lr.load_state_dict(*args, **kwargs)

    def get_last_lr(self, *args, **kwargs):
        return self.inner_lr.get_last_lr(*args, **kwargs)

    def get_lr(self, *args, **kwargs):
        return self.inner_lr.get_lr(*args, **kwargs)

    def print_lr(self, *args, **kwargs):
        return self.inner_lr.print_lr(*args, **kwargs)

    def step(self, *args, **kwargs):
        while self.steps < self.pl_module.global_step:
            self.inner_lr.step(*args, **kwargs)
            self.steps += 1


class GlobalSyncExponentialLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, lr_decay, global_step, verbose=False):
        self.global_step = global_step
        self.optimizer = optimizer
        self.lr_decay = lr_decay
        self.steps = 0
        last_epoch = -1
        super(GlobalSyncExponentialLR, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        current_lr_ratio = math.pow(self.lr_decay, self.global_step)
        lrs = [base_lr * current_lr_ratio for base_lr in self.base_lrs]
        return lrs

    def step(self, pl_module=None):
        if pl_module is not None:
            self.global_step = pl_module.global_step
