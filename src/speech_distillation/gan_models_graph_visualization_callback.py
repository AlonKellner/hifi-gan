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
            # sw.add_graph(pl_module.discriminator, wav)
            # h = pl_module.config
            # embedding = torch.zeros((h.embedding_size, h.segment_size // h.embedding_size))
            # embedding = embedding.cuda().unsqueeze(0)

            # for keeper in pl_module.keepers.values():
            #     sw.add_graph(keeper, embedding, use_strict_trace=False)
            # for hunter in pl_module.hunters.values():
            #     sw.add_graph(hunter, embedding, use_strict_trace=False)
