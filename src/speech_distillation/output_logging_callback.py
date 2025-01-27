from src.speech_distillation.output_sum_callback import OutputSumResetCallback
from logging_utils import rank


class OutputLoggingCallback(OutputSumResetCallback):

    def on_sum_reset(self, trainer, pl_module, batch_type, sums, amounts, global_step):
        sw = pl_module.logger.experiment
        self._log_recursive(
            logger=sw,
            prefix='{}_losses'.format(batch_type),
            sums=sums,
            amounts=amounts,
            log_index=global_step)

    def _log_recursive(self, logger, prefix, sums, amounts, log_index):
        if isinstance(sums, dict):
            for key, sum in sums.items():
                self._log_recursive(logger=logger, prefix=f'{prefix}/{key}', sums=sum, amounts=amounts,
                                    log_index=log_index)
        elif isinstance(sums, (list, tuple)):
            for key, sum in enumerate(sums):
                self._log_recursive(logger=logger, prefix=f'{prefix}/{key}', sums=sum, amounts=amounts,
                                    log_index=log_index)
        else:
            logger.add_scalar(rank(prefix), sums / amounts, log_index)
