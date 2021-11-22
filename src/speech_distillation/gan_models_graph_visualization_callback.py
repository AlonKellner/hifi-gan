import torch
from pytorch_lightning.callbacks import Callback

from src.utils import plot_spectrogram


class GanModelsGraphVisualizationCallback(Callback):
    def __init__(self):
        self.logged_graphs = False

    def on_validation_batch_start(
        self, trainer, pl_module, batch, batch_idx, dataloader_idx
    ) -> None:
        if not self.logged_graphs:
            self.logged_graphs = True
            wav, wav_path, time_labels, labels = batch
            wav = wav.unsqueeze(1)

            sw = pl_module.logger.experiment
            sw.add_graph(pl_module.generator, wav)
