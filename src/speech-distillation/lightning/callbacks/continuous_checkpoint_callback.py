from pytorch_lightning.callbacks import Callback

from pathlib import Path

from .utils import load_trainer_checkpoint, save_trainer_checkpoint


class ContinuousCheckpointCallback(Callback):
    def __init__(self, path, steps_interval):
        self.path = path
        self.steps_interval = steps_interval

    def on_init_start(self, trainer):
        if Path(self.path).exists():
            load_trainer_checkpoint(trainer, self.path)

    def on_batch_start(self, trainer, pl_module):
        if trainer.global_step % self.steps_interval == 0:
            save_trainer_checkpoint(trainer, self.path)
