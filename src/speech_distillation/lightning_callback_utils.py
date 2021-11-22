def load_trainer_checkpoint(trainer, path):
    if trainer.checkpoint_connector is not None and \
            trainer.checkpoint_connector.resume_checkpoint_path is None:
        trainer.checkpoint_connector.resume_checkpoint_path = path
    if trainer.resume_from_checkpoint is None:
        trainer.resume_checkpoint_path = path


def save_trainer_checkpoint(trainer, path):
    trainer.save_checkpoint(path)
