import torch
import torch.nn.functional as F
from pytorch_lightning.callbacks import Callback

from src.speech_distillation.output_sum_callback import OutputSumResetCallback


class ValidationClassificationCallback(Callback):
    def __init__(self, reset_interval, reset_callbacks: [OutputSumResetCallback] = []):
        self.reset_interval = reset_interval
        self.reset_callbacks: [OutputSumResetCallback] = reset_callbacks
        self.sums = {'soft': {}, 'one_hot': {}}
        self.amount = 0
        self.last_reset_step = -1

    def on_batch_start(self, trainer, pl_module) -> None:
        last_global_step = pl_module.global_step - 1
        should_reset = \
            last_global_step % self.reset_interval == 0 and \
            self.last_reset_step != last_global_step
        if should_reset:
            self.on_sum_reset(
                trainer,
                pl_module,
                'validation',
                self.sums,
                self.amount,
                last_global_step
            )

            self.sums = {'soft': {}, 'one_hot': {}}
            self.last_reset_step = last_global_step
            self.amount = 0

    def on_sum_reset(self, trainer, pl_module, batch_type, sums, amounts, global_step):
        for reset_callback in self.reset_callbacks:
            reset_callback.on_sum_reset(trainer, pl_module, batch_type, sums, amounts, global_step)

    def on_validation_batch_end(
            self,
            trainer,
            pl_module,
            outputs,
            batch,
            batch_idx: int,
            dataloader_idx: int,
    ):
        losses, outputs = outputs

        labels = outputs['label']
        truth = labels['truth']
        other_labels = {key: value for key, value in labels.items() if key != 'truth'}
        truth = self._recursive_one_hot(truth, next(iter(other_labels.values())))
        for key, value in other_labels.items():
            outer_soft = self._recursive_outer(truth, value)
            if key not in self.sums['soft'] or self.sums['soft'][key] is None:
                self.sums['soft'][key] = outer_soft
            else:
                self.sums['soft'][key] = self._recursively_add(self.sums['soft'][key], outer_soft)

            value_one_hot = self._recursive_one_hot(value, truth)
            outer_one_hot = self._recursive_outer(truth, value_one_hot)
            if key not in self.sums['one_hot'] or self.sums['one_hot'][key] is None:
                self.sums['one_hot'][key] = outer_one_hot
            else:
                self.sums['one_hot'][key] = self._recursively_add(self.sums['one_hot'][key], outer_one_hot)
        self.amount += 1

    def _recursively_add(self, loss_sum, outputs):
        if isinstance(loss_sum, dict):
            return {key: self._recursively_add(loss_sum[key], outputs[key]) for key in loss_sum.keys()}
        elif isinstance(loss_sum, list):
            return [self._recursively_add(sum_value, new_value) for sum_value, new_value in zip(loss_sum, outputs)]
        else:
            return loss_sum + outputs

    def _recursive_outer(self, truth, value):
        if isinstance(truth, dict):
            return {key: self._recursive_outer(truth[key], value[key]) for key in truth.keys()}
        elif isinstance(truth, list):
            return [self._recursive_outer(sum_value, new_value) for sum_value, new_value in zip(truth, value)]
        else:
            value = value.cpu().detach().squeeze()
            truth = truth.cpu().detach().squeeze()
            outer_sequence = torch.einsum('ai,bi->abi', truth, value)
            outer_sequence = outer_sequence.float()
            outer = outer_sequence.mean(dim=2)
            return outer

    def _recursive_one_hot(self, value, example):
        if isinstance(value, dict):
            return {key: self._recursive_one_hot(value[key], example[key]) for key in value.keys()}
        elif isinstance(value, list):
            return [self._recursive_one_hot(new_value, new_example) for new_value, new_example in zip(value, example)]
        else:
            value = value.cpu().detach().squeeze()
            if value.dtype not in [torch.int64, torch.int32, torch.int16, torch.int8]:
                value = value.argmax(dim=0)
            example = example.cpu().detach().squeeze()
            value = F.one_hot(value, num_classes=example.size(0)).transpose(0, 1)
            return value

