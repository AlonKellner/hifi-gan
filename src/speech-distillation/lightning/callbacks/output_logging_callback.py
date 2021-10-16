from pytorch_lightning.callbacks import Callback

from .utils import save_trainer_checkpoint


class OutputLoggingCallback(Callback):
    def __init__(self, logging_intervals):
        self.logging_intervals = logging_intervals
        self.loss_sums = {key: None for key in self.logging_intervals.keys()}
        self.loss_amounts = {key: 0 for key in self.logging_intervals.keys()}

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

        if loss_amount % self.logging_intervals[batch_type] == 0:
            pl_module.log('{}_losses'.format(batch_type), outputs)
            loss_sum = None
            loss_amount = 0

        self.loss_sums[batch_type] = loss_sum
        self.loss_amounts[batch_type] = loss_amount

    def _recursively_add(self, loss_sum, outputs):
        if isinstance(loss_sum, dict):
            return {key: self._recursively_add(loss_sum[key], outputs[key]) for key in loss_sum.keys()}
        elif isinstance(loss_sum, list):
            return [self._recursively_add(sum_value, new_value) for sum_value, new_value in zip(loss_sum, outputs)]
        else:
            return loss_sum + outputs

    def on_train_batch_end(self, *params) -> None:
        self._on_batch_end('train', *params)

    def on_validation_batch_end(self, *params) -> None:
        self._on_batch_end('validation', *params)

    def on_test_batch_end(self, *params) -> None:
        self._on_batch_end('test', *params)
