import os

from pytorch_lightning.callbacks import Callback

from .lightning_callback_utils import save_trainer_checkpoint


class HistoryCheckpointCallback(Callback):
    def __init__(self, steps_interval):
        self.steps_interval = steps_interval

    def on_batch_start(self, trainer, pl_module):
        checkpoint_dir = os.path.join(trainer.log_dir, 'checkpoints')
        if trainer.global_step % self.steps_interval == 0:
            step_path = os.path.join(checkpoint_dir, f'step_{trainer.global_step}')
            save_trainer_checkpoint(trainer, step_path)
