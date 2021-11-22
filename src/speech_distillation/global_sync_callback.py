import torch
from pytorch_lightning.callbacks import Callback


class GlobalSyncCallback(Callback):
    def __init__(self):
        self.num_sanity_val_steps = None
        self.val_check_interval = None

        self.last_val_step = 0
        self.trainer = None

    def on_init_end(self, trainer) -> None:
        self.trainer = trainer

        self.num_sanity_val_steps = trainer.num_sanity_val_steps
        self.val_check_interval = trainer.val_check_interval

        self._reset_trainer(trainer)

        trainer.fit_loop.epoch_loop._should_check_val_fx = self._should_check_val_fx

    def _should_check_val_fx(self, batch_idx: int, is_last_batch: bool) -> bool:
        if not self.trainer.enable_validation:
            return False

        is_val_check_epoch = (self.trainer.current_epoch + 1) % self.trainer.check_val_every_n_epoch == 0
        if not is_val_check_epoch:
            return False

        # val_check_batch is inf for iterable datasets with no length defined
        is_infinite_dataset = self.trainer.val_check_batch == float("inf")
        if is_last_batch and is_infinite_dataset:
            return True

        if self.trainer.should_stop:
            return True

        is_val_check_batch = is_last_batch
        if self.trainer.global_step == 0:
            is_val_check_batch = False
        elif isinstance(self.trainer.limit_train_batches, int) and is_infinite_dataset:
            is_val_check_batch = self.trainer.global_step % self.trainer.limit_train_batches == 0
        elif self.trainer.val_check_batch < 0:
            is_val_check_batch = False
        elif self.trainer.val_check_batch != float("inf"):
            is_val_check_batch = self.trainer.global_step % self.trainer.val_check_batch == 0
        return is_val_check_batch

    def _set_trainer(self, trainer):
        trainer.num_sanity_val_steps = self.num_sanity_val_steps
        trainer.val_check_interval = self.val_check_interval
        trainer.val_check_batch = self.val_check_interval

    def _reset_trainer(self, trainer):
        trainer.num_sanity_val_steps = 0
        trainer.val_check_interval = -1

    def on_train_start(self, trainer, pl_module) -> None:
        self._set_trainer(trainer)
        if pl_module.global_step <= 0:
            self._run_sanity(trainer)

    def _run_sanity(self, trainer):
        torch.set_grad_enabled(False)
        trainer.model.eval()
        trainer._run_sanity_check(trainer.lightning_module)
        trainer.model.train()
        torch.set_grad_enabled(True)
        model = trainer.lightning_module
        trainer.reset_train_val_dataloaders(model)
