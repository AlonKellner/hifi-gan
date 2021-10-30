from pytorch_lightning.callbacks import Callback


class GlobalSyncCallback(Callback):
    def __init__(self):
        self.num_sanity_val_steps = None
        self.limit_val_batches = None
        self.val_check_interval = None

        self.last_val_step = 0

    def on_init_end(self, trainer) -> None:
        self.num_sanity_val_steps = trainer.num_sanity_val_steps
        self.limit_val_batches = trainer.limit_val_batches
        self.val_check_interval = trainer.val_check_interval

        self._reset_trainer(trainer)

    def _set_trainer(self, trainer):
        trainer.num_sanity_val_steps = self.num_sanity_val_steps
        trainer.limit_val_batches = self.limit_val_batches
        trainer.val_check_interval = self.val_check_interval

    def _reset_trainer(self, trainer):
        trainer.num_sanity_val_steps = 0
        trainer.limit_val_batches = 0
        trainer.val_check_interval = 0

    def on_train_start(self, trainer, pl_module) -> None:
        if pl_module.global_step <= 0:
            self._set_trainer(trainer)
            trainer._run_sanity_check(pl_module)
            self._reset_trainer(trainer)

    def on_batch_start(self, trainer, pl_module) -> None:
        last_global_step = pl_module.global_step - 1
        should_val = \
            last_global_step % self.val_check_interval == 0 and \
            self.last_val_step != last_global_step
        if should_val:
            self._set_trainer(trainer)
            trainer.validate()
            self._reset_trainer(trainer)
            self.last_val_step = last_global_step

