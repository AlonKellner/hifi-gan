from pytorch_lightning.callbacks import Callback

from src.utils import plot_spectrogram


class GanValidationVisualizationCallback(Callback):
    def __init__(self, amount_to_log):
        self.amount_to_log = amount_to_log
        self.ground_truth_to_log = list(range(amount_to_log))
        self.predictions_to_log = list(range(amount_to_log))

    def on_validation_batch_start(
        self, trainer, pl_module, batch, batch_idx, dataloader_idx
    ) -> None:
        wav, wav_path, time_labels, labels = batch
        wav_generated = pl_module.generator(wav.unsqueeze(1))
        wav_diff = wav - wav_generated
        reconstruction_loss, mel_loss, wave_loss, wav_mel, wav_generated_mel = \
            pl_module.get_reconstruction_loss(wav, wav_generated)
        wav_diff_mel = pl_module.get_mel_spectrogram(wav_diff)

        wav_mel_diff_inverse = wav_mel - wav_generated_mel
        
        sw = pl_module.logger.experiment
        if batch_idx in self.ground_truth_to_log:
            self.ground_truth_to_log.remove(batch_idx)

            sw.add_audio('ground_truth/wav_{}'.format(batch_idx), wav[0], pl_module.global_step, pl_module.config.sampling_rate)
            sw.add_figure('ground_truth/wav_mel_{}'.format(batch_idx),
                          plot_spectrogram(wav_mel[0].cpu().numpy()),
                          pl_module.global_step)
            
        if batch_idx in self.predictions_to_log:
            sw.add_audio('generated/wav_{}'.format(batch_idx), wav_generated[0], pl_module.global_step,
                         pl_module.config.sampling_rate)
            sw.add_figure('generated/wav_mel_{}'.format(batch_idx),
                          plot_spectrogram(wav_generated_mel.squeeze(0).cpu().numpy()), pl_module.global_step)

            sw.add_audio('wave_diff/wav_{}'.format(batch_idx), wav_diff[0], pl_module.global_step,
                         pl_module.config.sampling_rate)
            sw.add_figure('wave_diff/wav_mel_{}'.format(batch_idx),
                          plot_spectrogram(wav_diff_mel.squeeze(0).cpu().numpy()), pl_module.global_step)
            sw.add_figure('mel_diff/wav_mel_{}'.format(batch_idx),
                          plot_spectrogram(wav_mel_diff_inverse.squeeze(0).cpu().numpy()), pl_module.global_step)
