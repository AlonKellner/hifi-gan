import torch
from pytorch_lightning.callbacks import Callback


class OutputSumResetCallback:
    def on_sum_reset(self, trainer, pl_module, batch_type, sums, amounts, global_step):
        raise NotImplementedError


class OutputSumCallback(Callback):
    def __init__(self, reset_intervals, reset_callbacks: [OutputSumResetCallback] = []):
        self.reset_intervals = reset_intervals
        self.reset_callbacks: [OutputSumResetCallback] = reset_callbacks
        self.loss_sums = {key: None for key in self.reset_intervals.keys()}
        self.loss_amounts = {key: 0 for key in self.reset_intervals.keys()}
        self.last_reset_steps = {key: -1 for key in self.reset_intervals.keys()}

    def on_batch_start(self, trainer, pl_module) -> None:
        last_global_step = pl_module.global_step - 1
        for batch_type, reset_interval in self.reset_intervals.items():
            last_reset_step = self.last_reset_steps[batch_type]
            should_reset = \
                last_global_step % reset_interval == 0 and \
                last_reset_step != last_global_step and \
                self.loss_sums[batch_type] is not None
            if should_reset:
                self.on_sum_reset(
                    trainer,
                    pl_module,
                    batch_type,
                    self.loss_sums[batch_type],
                    self.loss_amounts[batch_type],
                    last_global_step
                )
                self.last_reset_steps[batch_type] = last_global_step
                self.loss_sums[batch_type] = None
                self.loss_amounts[batch_type] = 0

    def on_sum_reset(self, trainer, pl_module, batch_type, sums, amounts, global_step):
        for reset_callback in self.reset_callbacks:
            reset_callback.on_sum_reset(trainer, pl_module, batch_type, sums, amounts, global_step)

    def _on_batch_end(
            self,
            batch_type,
            trainer,
            pl_module,
            outputs,
            batch,
            batch_idx: int,
            dataloader_idx: int,
    ):
        loss_sum = self.loss_sums[batch_type]
        loss_amount = self.loss_amounts[batch_type]
        if loss_sum is None:
            loss_sum = outputs
        else:
            loss_sum = self._recursively_add(loss_sum, outputs)
        loss_amount += 1

        self.loss_sums[batch_type] = loss_sum
        self.loss_amounts[batch_type] = loss_amount

    def _recursively_add(self, loss_sum, outputs):
        if isinstance(loss_sum, dict):
            return {key: self._recursively_add(loss_sum[key], outputs[key]) for key in loss_sum.keys()}
        elif isinstance(loss_sum, list):
            return [self._recursively_add(sum_value, new_value) for sum_value, new_value in zip(loss_sum, outputs)]
        else:
            if isinstance(outputs, torch.Tensor):
                outputs = outputs.item()
            return loss_sum + outputs

    def on_train_batch_end(self, *params) -> None:
        self._on_batch_end('train', *params)

    def on_validation_batch_end(self, *params) -> None:
        self._on_batch_end('validation', *params)

    def on_test_batch_end(self, *params) -> None:
        self._on_batch_end('test', *params)
