from pytorch_lightning.callbacks import Callback


class OutputLoggingCallback(Callback):
    def __init__(self, logging_intervals):
        self.logging_intervals = logging_intervals
        self.loss_sums = {key: None for key in self.logging_intervals.keys()}
        self.loss_amounts = {key: 0 for key in self.logging_intervals.keys()}
        self.last_log_steps = {key: -1 for key in self.logging_intervals.keys()}

    def on_batch_start(self, trainer, pl_module) -> None:
        last_global_step = pl_module.global_step - 1
        for batch_type, logging_interval in self.logging_intervals.items():
            last_log_step = self.last_log_steps[batch_type]
            should_log = \
                last_global_step % logging_interval == 0 and \
                last_log_step != last_global_step and \
                self.loss_sums[batch_type] is not None
            if should_log:
                sw = pl_module.logger.experiment
                self._log_recursive(
                    logger=sw,
                    prefix='{}_losses'.format(batch_type),
                    sums=self.loss_sums[batch_type],
                    amounts=self.loss_amounts[batch_type],
                    log_index=last_global_step)
                self.last_log_steps[batch_type] = last_global_step
                self.loss_sums[batch_type] = None
                self.loss_amounts[batch_type] = 0

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
            return loss_sum + outputs

    def on_train_batch_end(self, *params) -> None:
        self._on_batch_end('train', *params)

    def on_validation_batch_end(self, *params) -> None:
        self._on_batch_end('validation', *params)

    def on_test_batch_end(self, *params) -> None:
        self._on_batch_end('test', *params)

    def _log_recursive(self, logger, prefix, sums, amounts, log_index):
        if isinstance(sums, dict):
            for key, sum in sums.items():
                self._log_recursive(logger=logger, prefix=f'{prefix}/{key}', sums=sum, amounts=amounts,
                                    log_index=log_index)
        else:
            logger.add_scalar(prefix, sums / amounts, log_index)
