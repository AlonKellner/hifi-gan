import torch

from src.speech_distillation.output_sum_callback import OutputSumResetCallback
from src.utils import plot_matrix


class ConfusionLoggingCallback(OutputSumResetCallback):

    def on_sum_reset(self, trainer, pl_module, batch_type, sums, amounts, global_step):
        sw = pl_module.logger.experiment
        self._log_recursive(
            logger=sw,
            prefix='confusion',
            soft=sums['soft'],
            one_hot=sums['one_hot'],
            amounts=amounts,
            log_index=global_step)

    def _log_recursive(self, logger, prefix, soft, one_hot, amounts, log_index):
        if isinstance(soft, dict):
            for key in soft.keys():
                self._log_recursive(logger=logger,
                                    prefix=f'{prefix}/{key}',
                                    soft=soft[key],
                                    one_hot=one_hot[key],
                                    amounts=amounts,
                                    log_index=log_index)
        elif isinstance(soft, (list, tuple)):
            for key in range(len(soft)):
                self._log_recursive(logger=logger,
                                    prefix=f'{prefix}/{key}',
                                    soft=soft[key],
                                    one_hot=one_hot[key],
                                    amounts=amounts,
                                    log_index=log_index)
        else:
            soft_avg = soft / amounts
            one_hot_avg = one_hot / amounts
            avg_diff = one_hot_avg - soft_avg

            self._log_matrix(logger=logger, prefix=f'{prefix}/soft', matrix=soft_avg, log_index=log_index)
            self._log_matrix(logger=logger, prefix=f'{prefix}/one_hot', matrix=one_hot_avg, log_index=log_index)
            self._log_diff_matrix(logger=logger, prefix=f'{prefix}/diff', matrix=avg_diff, log_index=log_index)

    def _log_matrix(self, logger, prefix, matrix, log_index):
        recall = torch.trace(matrix).item()
        sum = matrix.sum().item()
        recall = recall/sum
        logger.add_scalar(f'{prefix}/recall', recall, log_index)

        logger.add_figure(prefix, plot_matrix(matrix.squeeze().cpu().numpy()), log_index)

    def _log_diff_matrix(self, logger, prefix, matrix, log_index):
        recall = torch.trace(matrix).item()
        recall = recall
        logger.add_scalar(f'{prefix}/recall', recall, log_index)

        logger.add_figure(prefix, plot_matrix(matrix.squeeze().cpu().numpy()), log_index)


