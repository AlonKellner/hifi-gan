from pytorch_lightning.callbacks import Callback
from torch.nn.utils import clip_grad_norm_


class ManualOptimizationCallback(Callback):
    def __init__(self, accumulated_grad_batches=1, clip_value=1000.0, optimizer_args=(), scheduler_args=()):
        self.accumulated_grad_batches = accumulated_grad_batches
        self.clip_value = clip_value
        self.optimizer_args = optimizer_args
        self.scheduler_args = scheduler_args

    def on_train_start(self, trainer, pl_module) -> None:
        pl_module.automatic_optimization = False

    def on_batch_end(self, trainer, pl_module) -> None:
        should_step_gradient = pl_module.global_step % self.accumulated_grad_batches == 0
        if should_step_gradient:
            sw = pl_module.logger.experiment
            learning_models = pl_module.get_learning_models()
            if self.clip_value > 0:
                for key, learning_model in learning_models.items():
                    norm = clip_grad_norm_(learning_model.parameters(), self.clip_value)
                    sw.add_scalar(f'params/grad_clip/norm/{key}', norm, pl_module.global_step)
            optimizers = pl_module.optimizers()
            optimizers = optimizers if isinstance(optimizers, list) else [optimizers]
            for optimizer in optimizers:
                optimizer.step(*self.optimizer_args)
                optimizer.zero_grad()

            learning_models_keys = list(learning_models.keys())
            schedulers = pl_module.lr_schedulers()
            schedulers = schedulers if isinstance(schedulers, list) else [schedulers]
            for index, scheduler in enumerate(schedulers):
                current_model_name = learning_models_keys[index]
                scheduler.step(*self.scheduler_args)
                for index2, lr in enumerate(scheduler.get_lr()):
                    sw.add_scalar(f'params/lr/{current_model_name}/{index2}', lr, pl_module.global_step)
