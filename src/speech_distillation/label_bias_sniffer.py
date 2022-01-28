import warnings

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

from src.speech_distillation.best_checkpoint_callback import BestCheckpointCallback
from src.speech_distillation.custom_losses import recursive_loss
from src.speech_distillation.global_sync_callback import GlobalSyncCallback
from src.speech_distillation.global_sync_lr_scheduler import GlobalSyncDecoratorLR
from src.speech_distillation.output_sum_callback import OutputSumCallback

warnings.simplefilter(action='ignore', category=FutureWarning)
import json
import torch
from torch.utils.data import DistributedSampler, DataLoader
from src.env import AttrDict
from src.speech_distillation.continuous_checkpoint_callback import ContinuousCheckpointCallback
from src.speech_distillation.history_checkpoint_callback import HistoryCheckpointCallback
from src.speech_distillation.output_logging_callback import OutputLoggingCallback
from src.speech_distillation.manual_optimization_callback import ManualOptimizationCallback

from static_configs import generate_sniffers_configs_by_example
from configurable_module import get_module_from_config
from multilabel_wave_dataset import MultilabelWaveDataset

from torchsummary import summary, RandInt, InputSize

torch.backends.cudnn.benchmark = True


class LabelBiasSniffer(pl.LightningModule):
    def __init__(self, sniffers, sniffer_key, config=None):
        super().__init__()
        self.sniffers = sniffers  # TODO: DELETE!
        self.sniffer_key = sniffer_key
        self.sniffer = sniffers[self.sniffer_key]
        self.loss_func = torch.nn.CrossEntropyLoss()
        if config is None:
            config = {
                'learning_rate': 0.0001,
                'lr_decay': 0.9999,
                'adam_b1': 0.8,
                'adam_b2': 0.99
            }
        self.learning_rate = config['learning_rate']
        self.adam_b1 = config['adam_b1']
        self.adam_b2 = config['adam_b2']
        self.lr_decay = config['lr_decay']

    def forward(self, x):
        return self.sniffer(x)

    def configure_optimizers(self):
        optim = torch.optim.AdamW(
            self.sniffer.parameters(),
            self.learning_rate,
            betas=(self.adam_b1, self.adam_b2),
            amsgrad=True
        )

        scheduler = GlobalSyncDecoratorLR(
            self,
            torch.optim.lr_scheduler.ExponentialLR(optim, gamma=self.lr_decay)
        )
        return [optim], [scheduler]

    def training_step(self, train_batch, batch_idx):
        loss = self.get_loss(train_batch)
        self.manual_backward(loss)
        return loss

    def validation_step(self, validation_batch, batch_idx):
        return self.get_loss(validation_batch)

    def get_loss(self, batch):
        wav, wav_path, time_labels, labels = batch

        other_groups_predictions, other_groups_predictions_var = self.sniffer(time_labels[self.sniffer_key])
        loss = recursive_loss(self.loss_func, other_groups_predictions, time_labels)

        return loss


def generate_sniffers_by_example(example_item, cache_hook=lambda k, x: x(), layers=[], one_hot=False):
    sniffers_configs = generate_sniffers_configs_by_example(example_item, layers=layers, one_hot=one_hot)
    return torch.nn.ModuleDict(
        {key: get_module_from_config(cache_hook(key, lambda: sniffer_config)) for key, sniffer_config in
         sniffers_configs.items()})


def main():
    print('Initializing Training Process...')

    with open('config/config.json') as f:
        data = f.read()

    config_dict = json.loads(data)
    h = AttrDict(config_dict)

    batch_size = 5

    train_dataset = MultilabelWaveDataset(
        data_dir='/datasets',
        cache_dir='/datasets/training_audio',
        name='train',
        config_path='**/train_data_config/*.json',
        segment_length=h.segment_length,
        sampling_rate=h.sampling_rate,
        embedding_size=h.embedding_size,
        augmentation_config=config_dict['augmentation'],
        disable_wavs=True
    )

    validation_dataset = MultilabelWaveDataset(
        data_dir='/datasets',
        cache_dir='/datasets/training_audio',
        name='train',
        config_path='**/train_data_config/*.json',
        segment_length=h.segment_length,
        sampling_rate=h.sampling_rate,
        embedding_size=h.embedding_size,
        augmentation_config=config_dict['augmentation'],
        deterministic=True,
        size=100,
        disable_wavs=True
    )

    test_dataset = MultilabelWaveDataset(
        data_dir='/datasets',
        cache_dir='/datasets/training_audio',
        name='test',
        config_path='**/test_data_config/*.json',
        segment_length=h.segment_length,
        sampling_rate=h.sampling_rate,
        embedding_size=h.embedding_size,
        deterministic=True,
        augmentation_config=config_dict['augmentation'],
        disable_wavs=True
    )
    train_sampler = DistributedSampler(train_dataset) if h.num_gpus > 1 else None

    train_loader = DataLoader(train_dataset, num_workers=h.num_workers, shuffle=True,
                              sampler=train_sampler,
                              batch_size=batch_size,
                              pin_memory=True,
                              drop_last=True)

    validation_loader = DataLoader(validation_dataset, num_workers=h.num_workers, shuffle=False,
                                   sampler=None,
                                   batch_size=batch_size,
                                   pin_memory=True,
                                   drop_last=True)

    example_item = train_dataset.label_option_groups
    sniffers = generate_sniffers_by_example(example_item)
    for key, sniffer in sniffers.items():
        print('{} sniffer:'.format(key))
        input_size = InputSize(
            {label: (h.segment_length // h.embedding_size,) for label, value in example_item[key].items()})
        summary(sniffer,
                input_size=input_size,
                dtypes={label: RandInt(type=torch.LongTensor, high=value)
                        for label, value in example_item[key].items()},
                device='cpu')

    # model
    models = {key: LabelBiasSniffer(sniffers, sniffer_key=key, config={
        'learning_rate': 0.0001,
        'lr_decay': 0.9999,
        'adam_b1': 0.8,
        'adam_b2': 0.99
    }) for key in sniffers.keys()}

    # training
    experiment_name = 'default'
    version = 1
    accumulated_grad = 1000 / batch_size
    intervals = {
        'train': 1000 / batch_size,
        'validation': 1000 / batch_size
    }
    trainers = {
        key: create_trainer(f'/mount/sniffers/logs/{key}', experiment_name, version, intervals, accumulated_grad)
        for key in models.keys()
    }
    for key, model in models.items():
        trainers[key].fit(model, train_loader, validation_loader)


def create_trainer(log_dir, experiment_name, version, intervals, accumulated_grad):
    best_checkpoint_callback = BestCheckpointCallback()
    return pl.Trainer(
        gpus=1,
        num_nodes=1,
        precision=32,
        max_epochs=1,
        logger=pl_loggers.TensorBoardLogger(
            log_dir,
            name=experiment_name,
            version=version,
            default_hp_metric=False),
        val_check_interval=intervals['validation'],
        num_sanity_val_steps=2,
        callbacks=[
            GlobalSyncCallback(),
            HistoryCheckpointCallback(),
            ContinuousCheckpointCallback(),
            best_checkpoint_callback,
            ManualOptimizationCallback(accumulated_grad),
            OutputSumCallback(
                intervals,
                reset_callbacks=[
                    OutputLoggingCallback(),
                    best_checkpoint_callback
                ]
            )
        ]
    )


if __name__ == '__main__':
    main()
