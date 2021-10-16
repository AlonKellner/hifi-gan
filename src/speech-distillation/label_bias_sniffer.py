import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

import warnings
import warnings

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

warnings.simplefilter(action='ignore', category=FutureWarning)
import json
import torch
from torch.utils.data import DistributedSampler, DataLoader
from src.env import AttrDict
from lightning.callbacks.continuous_checkpoint_callback import ContinuousCheckpointCallback
from lightning.callbacks.history_checkpoint_callback import HistoryCheckpointCallback
from lightning.callbacks.output_logging_callback import OutputLoggingCallback

from torchsummary import summary, InputSize, RandInt

from static_configs import generate_sniffers_configs_by_example
from configurable_module import get_module_from_config
from multilabel_wave_dataset import MultilabelDataset

torch.backends.cudnn.benchmark = True


class LabelBiasSniffer(pl.LightningModule):
    def __init__(self, sniffers, sniffer_key, config):
        super().__init__()
        self.sniffers = sniffers
        self.sniffer_key = sniffer_key
        self.sniffer = self.sniffers[self.sniffer_key]
        self.loss = torch.nn.CrossEntropyLoss()
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

        scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=self.lr_decay)
        return [optim], [scheduler]

    def training_step(self, train_batch, batch_idx):
        labels = train_batch
        other_groups_predictions = self.sniffer(labels[self.sniffer_key])

        loss = 0
        for other_group_key, other_group_prediction in other_groups_predictions.items():
            group_loss = 0
            for other_group_label, other_group_label_prediction in other_group_prediction.items():
                current_loss = self.loss(other_group_label_prediction, labels[other_group_key][other_group_label])
                group_loss = current_loss + group_loss
            loss = group_loss + loss

        return loss


def generate_sniffers_by_example(example_item):
    sniffers_configs = generate_sniffers_configs_by_example(example_item)
    return torch.nn.ModuleDict({key: get_module_from_config(sniffer_config) for key, sniffer_config in sniffers_configs.items()})


def main():
    print('Initializing Training Process..')

    with open('config/config_none.json') as f:
        data = f.read()

    json_config = json.loads(data)
    h = AttrDict(json_config)

    train_dataset = MultilabelDataset(
        dir='/datasets/training_audio',
        name='train_dataset_files',
        config_path='**/train_data_config/*.json',
        segment_size=h.segment_size,
        sampling_rate=h.sampling_rate,
        embedding_size=3 * 7 * 13,
    )

    test_dataset = MultilabelDataset(
        dir='/datasets/training_audio',
        name='test_dataset_files',
        config_path='**/test_data_config/*.json',
        segment_size=h.segment_size,
        sampling_rate=h.sampling_rate,
        embedding_size=3 * 7 * 13,
    )

    train_sampler = DistributedSampler(train_dataset) if h.num_gpus > 1 else None
    test_sampler = DistributedSampler(test_dataset) if h.num_gpus > 1 else None
    batch_size = 16
    train_loader = DataLoader(train_dataset, num_workers=4, shuffle=True,
                        sampler=train_sampler,
                        batch_size=batch_size,
                        pin_memory=True,
                        drop_last=True)
    test_loader = DataLoader(test_dataset, num_workers=4, shuffle=True,
                        sampler=test_sampler,
                        batch_size=batch_size,
                        pin_memory=True,
                        drop_last=True)
    example_item = train_dataset.label_option_groups
    sniffers = generate_sniffers_by_example(example_item)
    # for key, sniffer in sniffers.items():
    #     print('{} sniffer:'.format(key))
    #     input_size = InputSize({label: (h.segment_size,) for label, value in example_item[key].items()})
    #     summary(sniffer,
    #             input_size=input_size,
    #             dtypes={label: RandInt(type=torch.LongTensor, high=value)
    #                     for label, value in example_item[key].items()},
    #             batch_size=batch_size,
    #             device='cpu')

    # model
    models = {key: LabelBiasSniffer(sniffers, sniffer_key=key, config={
        'learning_rate': 0.075,
        'lr_decay': 0.99,
        'adam_b1': 0.8,
        'adam_b2': 0.99
    }) for key in sniffers.keys()}
    # model = LitModel(generator, 0.0002, 0.8, 0.99)

    # training
    experiment_name = 'default'

    trainers = {key: pl.Trainer(
        gpus=1,
        num_nodes=1,
        precision=32,
        auto_lr_find='learning_rate',
        max_epochs=1,
        logger=pl_loggers.TensorBoardLogger('/mount/label_bias_sniffer/logs/{}'.format(key),
                                            name=experiment_name),
        callbacks=[
            ContinuousCheckpointCallback('/mount/label_bias_sniffer/checkpoints/{}/{}/latest'.format(key, experiment_name), 100),
            HistoryCheckpointCallback('/mount/label_bias_sniffer/checkpoints/{}/{}/step'.format(key, experiment_name), 5000),
            OutputLoggingCallback({
                'train': 100
            })
        ]
    ) for key in models.keys()}
    for key, model in models.items():
        # result = trainers[key].tune(model, loader)
        # print('~ {} ~'.format(key))
        # print('best lr: {}'.format(result['lr_find'].suggestion()))
        trainers[key].fit(model, train_loader)


if __name__ == '__main__':
    main()
