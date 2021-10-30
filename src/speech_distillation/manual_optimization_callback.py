from pytorch_lightning.callbacks import Callback


class ManualOptimizationCallback(Callback):
    def __init__(self, accumulated_grad_batches):
        self.accumulated_grad_batches = accumulated_grad_batches

    def on_batch_end(self, trainer, pl_module) -> None:
        should_step_gradient = pl_module.global_step % self.accumulated_grad_batches == 0
        if should_step_gradient:
            optimizers = pl_module.optimizers()
            for optimizer in optimizers:
                optimizer.step()
                optimizer.zero_grad()

            schedulers = pl_module.lr_schedulers()
            for scheduler in schedulers:
                scheduler.step()
