from pathlib import Path


def load_trainer_checkpoint(trainer, path):
    if trainer.checkpoint_connector.resume_checkpoint_path is None:
        trainer.checkpoint_connector.resume_checkpoint_path = path


def save_trainer_checkpoint(trainer, path):
    trainer.save_checkpoint(path)
