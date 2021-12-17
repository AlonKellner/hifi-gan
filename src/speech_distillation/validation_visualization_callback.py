import torch
from pytorch_lightning.callbacks import Callback

from src.utils import plot_spectrogram, plot_categorical
from logging_utils import rank


class ValidationVisualizationCallback(Callback):
    def __init__(self, amounts_to_log):
        self.amounts_to_log = amounts_to_log
        self.truth_to_log = {}
        self.to_log = {}

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

        sw = pl_module.logger.experiment
        self.visualize(trainer, pl_module, batch, batch_idx, dataloader_idx, sw,
                       'wavs', self._visualize_wav, 'few', outputs['wav'])
        self.visualize(trainer, pl_module, batch, batch_idx, dataloader_idx, sw,
                       'mels', self._visualize_mel, 'few', outputs['mel'])
        self.visualize(trainer, pl_module, batch, batch_idx, dataloader_idx, sw,
                       'labels', self._visualize_label, 'few', outputs['label'], level=1)
        self.visualize(trainer, pl_module, batch, batch_idx, dataloader_idx, sw,
                       'outputs', self._visualize_output, 'once', outputs)
        del outputs
        self.visualize_model_parameters(trainer, pl_module, batch, batch_idx, dataloader_idx,
                                        sw)

    def visualize_model_parameters(self, trainer, pl_module, batch, batch_idx, dataloader_idx, sw):
        if self._check_to_log(batch_idx, 'parameters', 'once'):
            models = pl_module.get_learning_models()
            for name, model in models.items():
                sw.add_histogram(rank(f'parameters/{name}'), torch.cat([p.detach().view(-1) for p in model.parameters()]),
                                 pl_module.global_step)

    def visualize(self, trainer, pl_module, batch, batch_idx, dataloader_idx, sw, prefix, visualize, log_type, data,
                  level=1000):
        if self._check_to_log(batch_idx, prefix, log_type):
            self.visualize_recursive(
                logger=sw,
                pl_module=pl_module,
                prefix=prefix,
                visualize=visualize,
                batch_idx=batch_idx,
                log_type=log_type,
                data=data,
                level=level
            )

    def visualize_recursive(self, logger, pl_module, batch_idx, prefix, data, visualize, log_type, level):
        if isinstance(data, dict) and level > 0:
            for key, value in data.items():
                new_prefix = f'{prefix}/{key}'
                if key != 'truth' or self._check_truth_to_log(batch_idx, new_prefix, log_type):
                    self.visualize_recursive(
                        logger=logger,
                        pl_module=pl_module,
                        batch_idx=batch_idx,
                        prefix=new_prefix,
                        log_type=log_type,
                        visualize=visualize,
                        data=value,
                        level=level-1
                    )
        elif isinstance(data, (list, tuple)) and level > 0:
            for key, value in enumerate(data):
                self.visualize_recursive(
                    logger=logger,
                    pl_module=pl_module,
                    batch_idx=batch_idx,
                    prefix=f'{prefix}/{key}',
                    log_type=log_type,
                    visualize=visualize,
                    data=value,
                    level=level-1
                )
        else:
            visualize(logger, pl_module, batch_idx, f'{prefix}/{batch_idx}', data)

    def _check_truth_to_log(self, index, key, log_type):
        truth_to_log = self._get_truth_to_log(key, log_type)
        if index in truth_to_log:
            truth_to_log.remove(index)
            return True
        return False

    def _get_truth_to_log(self, key, log_type):
        if key not in self.truth_to_log:
            self.truth_to_log[key] = list(range(self.amounts_to_log[log_type]))
        return self.truth_to_log[key]

    def _check_to_log(self, index, key, log_type):
        truth_to_log = self._get_truth_to_log(key, log_type)
        return index in truth_to_log

    def _get_to_log(self, key, log_type):
        if key not in self.truth_to_log:
            self.truth_to_log[key] = list(range(self.amounts_to_log[log_type]))
        return self.truth_to_log[key]

    def _visualize_wav(self, sw, pl_module, batch_idx, prefix, wav):
        for index, sub_wav in enumerate(wav):
            sw.add_audio(rank(f'{prefix}/{index}'), sub_wav.cpu().numpy(), pl_module.global_step,
                         pl_module.config.sampling_rate)

    def _visualize_mel(self, sw, pl_module, batch_idx, prefix, mel):
        for index, sub_mel in enumerate(mel):
            sw.add_figure(rank(f'{prefix}/{index}'), plot_spectrogram(sub_mel.cpu().numpy()),
                          pl_module.global_step)

    def _visualize_label(self, sw, pl_module, batch_idx, prefix, label):
        cat_label = self._cat_recursive(label)
        for index, sub_label in enumerate(cat_label):
            sw.add_figure(rank(f'{prefix}/{index}'), plot_categorical(sub_label.squeeze().cpu().numpy()),
                          pl_module.global_step)

    def _visualize_output(self, sw, pl_module, batch_idx, prefix, output):
        sw.add_histogram(rank(prefix), output, pl_module.global_step)

    def _cat_recursive(self, label):
        if isinstance(label, dict):
            label_list = list(label.items())
            label_sorted = list(sorted(label_list, key=lambda pair: pair[0]))
            values = [self._cat_recursive(value) for key, value in label_sorted]
            return torch.cat(values, dim=1)
        else:
            label = label.squeeze()
            if label.dtype not in [torch.int64, torch.int32, torch.int16, torch.int8]:
                label = label.argmax(dim=1)
            label = label.squeeze().unsqueeze(1)
            return label
