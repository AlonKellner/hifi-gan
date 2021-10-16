from pytorch_lightning.callbacks import Callback

from .utils import save_trainer_checkpoint


class HistoryCheckpointCallback(Callback):
    def __init__(self, path, steps_interval):
        self.path = path
        self.steps_interval = steps_interval

    def on_batch_start(self, trainer, pl_module):
        if trainer.global_step % self.steps_interval == 0:
            save_trainer_checkpoint(trainer, '%s_%d' % (self.path, trainer.global_step))
