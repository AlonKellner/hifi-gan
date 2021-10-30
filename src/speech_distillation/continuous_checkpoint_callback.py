import os

from pytorch_lightning.callbacks import Callback

from pathlib import Path

from .lightning_callback_utils import load_trainer_checkpoint, save_trainer_checkpoint


class ContinuousCheckpointCallback(Callback):
    def __init__(self, steps_interval):
        self.steps_interval = steps_interval
        self.latest_path = None

    def on_init_end(self, trainer):
        checkpoint_dir = os.path.join(trainer.log_dir, 'checkpoints')
        self.latest_path = os.path.join(checkpoint_dir, f'latest')
        if Path(self.latest_path).exists():
            load_trainer_checkpoint(trainer, self.latest_path)

    def on_batch_start(self, trainer, pl_module):
        if trainer.global_step % self.steps_interval == 0:
            save_trainer_checkpoint(trainer, self.latest_path)
