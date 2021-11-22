import os
from pathlib import Path

import torch
import yaml
from pytorch_lightning import Callback
from yaml import FullLoader

from src.speech_distillation.lightning_callback_utils import save_trainer_checkpoint
from src.speech_distillation.output_sum_callback import OutputSumResetCallback


class BestCheckpointCallback(OutputSumResetCallback, Callback):
    def __init__(self, checkpoint_threshold=float('inf')):
        self.current_best = checkpoint_threshold
        self.best_score_path = None

    def on_init_end(self, trainer):
        checkpoint_dir = os.path.join(trainer.log_dir, 'checkpoints')
        self.best_score_path = os.path.join(checkpoint_dir, f'best_score.yaml')
        if Path(self.best_score_path).exists():
            with open(self.best_score_path, 'r') as cache:
                self.current_best = yaml.load(cache, FullLoader)

    def on_sum_reset(self, trainer, pl_module, batch_type, sums, amounts, global_step):
        if batch_type == 'validation':
            new_avg = self._avg_total_recursive(sums, amounts)
            if new_avg < self.current_best:
                self.current_best = new_avg
                if isinstance(self.current_best, torch.Tensor):
                    self.current_best = self.current_best.item()
                self._save_best(trainer, pl_module, self.current_best)

    def _save_best(self, trainer, pl_module, new_best_score):
        checkpoint_dir = os.path.join(trainer.log_dir, 'checkpoints')
        best_path = os.path.join(checkpoint_dir, f'best')
        save_trainer_checkpoint(trainer, best_path)
        with open(self.best_score_path, 'w') as cache:
            yaml.dump(new_best_score, cache)

    def _avg_total_recursive(self, sums, amounts):
        if isinstance(sums, dict):
            return sum(self._avg_total_recursive(sums=sub_sum, amounts=amounts) for key, sub_sum in sums.items())
        else:
            return sums / amounts
