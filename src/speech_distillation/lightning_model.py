import warnings

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from torch.nn import functional as F

from src.speech_distillation.gan_models_graph_visualization_callback import \
    GanModelsGraphVisualizationCallback
from src.speech_distillation.global_sync_callback import GlobalSyncCallback

warnings.simplefilter(action='ignore', category=FutureWarning)
import argparse
import json
import torch
from torch.utils.data import DistributedSampler, DataLoader
from src.env import AttrDict, build_env
from multilabel_wave_dataset import MultilabelWaveDataset
from src.meldataset import mel_spectrogram
from src.speech_distillation.continuous_checkpoint_callback import ContinuousCheckpointCallback
from src.speech_distillation.history_checkpoint_callback import HistoryCheckpointCallback
from src.speech_distillation.output_logging_callback import OutputLoggingCallback
from src.speech_distillation.manual_optimization_callback import ManualOptimizationCallback
from src.speech_distillation.valve_decay_callback import ValveDecayCallback
from src.speech_distillation.gan_validation_visualization_callback import \
    GanValidationVisualizationCallback

from torchsummary import summary

from static_configs import get_static_generator_config, \
    get_static_all_in_one_discriminator
from configurable_module import get_module_from_config
from custom_losses import feature_loss, one_loss, zero_loss

torch.backends.cudnn.benchmark = True

from embedding_classifiers.embedding_classifiers_static_configs import generate_keepers_by_example, \
    generate_hunters_by_example


class GanAutoencoder(pl.LightningModule):
    def __init__(self, generator, discriminator, keepers, hunters, config, args):
        super().__init__()
        self.automatic_optimization = False
        self.accumulated_grad_batches = config.accumulated_grad_batches
        self.gen_learning_rate = config.gen_learning_rate
        self.gen_adam_b1 = config.gen_adam_b1
        self.gen_adam_b2 = config.gen_adam_b2
        self.disc_learning_rate = config.disc_learning_rate
        self.disc_adam_b1 = config.disc_adam_b1
        self.disc_adam_b2 = config.disc_adam_b2
        self.generator = generator
        self.discriminator = discriminator
        self.keepers = keepers
        self.hunters = hunters
        self.config = config
        self.args = args

        self.valves_config = self.config.valves

    def forward(self, x):
        return self.generator(x)

    def configure_optimizers(self):
        optim_g = torch.optim.AdamW(
            self.generator.parameters(),
            self.gen_learning_rate,
            betas=(self.gen_adam_b1, self.gen_adam_b2),
            amsgrad=True
        )
        optim_d = torch.optim.AdamW(
            self.discriminator.parameters(),
            self.disc_learning_rate,
            betas=(self.disc_adam_b1, self.disc_adam_b2),
            amsgrad=True
        )

        scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=self.config.lr_decay)
        scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=self.config.lr_decay)

        return [optim_g, optim_d], \
               [scheduler_g, scheduler_d]

    def training_step(self, train_batch, batch_idx):
        wav, wav_path, time_labels, labels = train_batch
        wav = wav.unsqueeze(1)

        wav_generated = self.generator(wav)
        gen_current_loss = 0
        disc_current_loss = 0

        reconstruction_loss, mel_loss, wave_loss, _, _ = self.get_reconstruction_loss(wav, wav_generated)
        gen_current_loss = reconstruction_loss + gen_current_loss

        adversarial_loss, disc_loss, fmap_loss = self.get_adversarial_loss(wav, wav_generated, self.discriminator)
        gen_current_loss = adversarial_loss + gen_current_loss

        discrimination_loss, real_loss, fake_loss = self.get_discrimination_loss(wav, wav_generated, self.discriminator)
        disc_current_loss = discrimination_loss + disc_current_loss

        self.manual_backward(gen_current_loss)
        self.manual_backward(disc_current_loss)

        losses_dict = {
            'generator': {
                'total': gen_current_loss,
                'reconstruction': {
                    'mel': mel_loss,
                    'wave': wave_loss
                },
                'adversarial': {
                    'disc': disc_loss,
                    'fmap': fmap_loss
                }
            },
            'disriminator': {
                'total': disc_current_loss,
                'adversarial': {
                    'real': real_loss,
                    'fake': fake_loss
                }
            }
        }
        losses_dict = self._detach_recursively(losses_dict)
        return losses_dict

    def _detach_recursively(self, losses):
        if isinstance(losses, dict):
            return {key: self._detach_recursively(loss) for key, loss in losses.items()}
        else:
            return losses.detach()

    def validation_step(self, val_batch, batch_idx):
        wav, wav_path, time_labels, labels = val_batch
        wav_generated = self.generator(wav.unsqueeze(1))
        reconstruction_loss, mel_loss, wave_loss, wav_mel, wav_generated_mel = \
            self.get_reconstruction_loss(wav, wav_generated)

        return {
            'wave': wave_loss,
            'mel': mel_loss
        }

    def get_mel_spectrogram(self, wav):
        return mel_spectrogram(
            wav.squeeze(1),
            self.config.n_fft,
            self.config.num_mels,
            self.config.sampling_rate,
            self.config.hop_size,
            self.config.win_size,
            self.config.fmin,
            self.config.fmax_for_loss)

    def get_reconstruction_loss(self, wav, wav_generated):
        wav_mel = self.get_mel_spectrogram(wav)
        wav_generated_mel = self.get_mel_spectrogram(wav_generated)
        loss_mel = F.l1_loss(wav_mel, wav_generated_mel) * 10
        loss_wave = F.l1_loss(wav, wav_generated) * 350
        loss_recon = (loss_mel + loss_wave)
        return loss_recon, loss_mel, loss_wave, wav_mel, wav_generated_mel

    def get_discrimination_loss(self, wav, wav_generated, discriminator):
        wav_mom_r, wav_fmap_r = discriminator(wav)
        wav_all_r, wav_all_var_r = wav_mom_r
        wav_d_r, wav_sub_d_r = wav_all_r
        wav_mom_g_detach, wav_fmap_g_detach = discriminator(wav_generated.detach())
        wav_all_g_detach, wav_all_var_g_detach = wav_mom_g_detach
        wav_d_g_detach, wav_sub_d_g_detach = wav_all_g_detach

        loss_disc_r = one_loss(wav_d_r)
        loss_sub_disc_r = sum(one_loss(sub_d) for sub_d in wav_sub_d_r)
        loss_all_disc_r = loss_disc_r + loss_sub_disc_r

        loss_disc_g = zero_loss(wav_d_g_detach)
        loss_sub_disc_g = sum(zero_loss(sub_d) for sub_d in wav_sub_d_g_detach)
        loss_all_disc_g = loss_disc_g + loss_sub_disc_g

        loss_all_disc = loss_all_disc_r + loss_all_disc_g
        return loss_all_disc, loss_all_disc_r, loss_all_disc_g

    def get_adversarial_loss(self, wav, wav_generated, discriminator):
        wav_mom_r, wav_fmap_r = discriminator(wav)
        wav_mom_g, wav_fmap_g = discriminator(wav_generated)
        wav_all_g, wav_all_var_g = wav_mom_g
        wav_d_g, wav_sub_d_g = wav_all_g

        loss_disc = one_loss(wav_d_g)
        loss_sub_disc = sum(one_loss(sub_d) for sub_d in wav_sub_d_g)
        loss_all_disc = (loss_disc + loss_sub_disc) * 0.01

        loss_fm = feature_loss(wav_fmap_r, wav_fmap_g) * 0.1

        loss_adv = (loss_all_disc + loss_fm)
        return loss_adv, loss_all_disc, loss_fm


def main():
    print('Initializing Training Process..')

    parser = argparse.ArgumentParser()

    parser.add_argument('--group_name', default=None)
    parser.add_argument('--input_wavs_dir', default='LJSpeech-1.1/wavs')
    parser.add_argument('--input_mels_dir', default='ft_dataset')
    parser.add_argument('--input_training_file', default='LJSpeech-1.1/training.txt')
    parser.add_argument('--input_validation_file', default='LJSpeech-1.1/validation.txt')
    parser.add_argument('--checkpoint_path', default='cp_hifigan')
    parser.add_argument('--config', default='')
    parser.add_argument('--training_epochs', default=3100, type=int)
    parser.add_argument('--stdout_interval', default=5, type=int)
    parser.add_argument('--checkpoint_interval', default=5000, type=int)
    parser.add_argument('--summary_interval', default=100, type=int)
    parser.add_argument('--validation_interval', default=1000, type=int)
    parser.add_argument('--fine_tuning', default=False, type=bool)

    a = parser.parse_args()

    with open(a.config) as f:
        data = f.read()

    json_config = json.loads(data)
    h = AttrDict(json_config)
    build_env(a.config, 'config.json', a.checkpoint_path)

    train_dataset = MultilabelWaveDataset(
        base_dir='/datasets',
        dir='/datasets/training_audio',
        name='train',
        config_path='**/train_data_config/*.json',
        segment_size=h.segment_size,
        sampling_rate=h.sampling_rate,
        embedding_size=h.embedding_size,
        augmentation_config=json_config['augmentation']
    )

    validation_dataset = MultilabelWaveDataset(
        base_dir='/datasets',
        dir='/datasets/training_audio',
        name='train',
        config_path='**/train_data_config/*.json',
        segment_size=h.segment_size,
        sampling_rate=h.sampling_rate,
        embedding_size=h.embedding_size,
        augmentation_config=json_config['augmentation'],
        deterministic=True,
        size=100
    )

    test_dataset = MultilabelWaveDataset(
        base_dir='/datasets',
        dir='/datasets/training_audio',
        name='test',
        config_path='**/test_data_config/*.json',
        segment_size=h.segment_size,
        sampling_rate=h.sampling_rate,
        embedding_size=h.embedding_size,
        deterministic=True,
        augmentation_config=json_config['augmentation']
    )
    train_sampler = DistributedSampler(train_dataset) if h.num_gpus > 1 else None

    train_loader = DataLoader(train_dataset, num_workers=h.num_workers, shuffle=True,
                              sampler=train_sampler,
                              batch_size=h.batch_size,
                              pin_memory=True,
                              drop_last=True)

    validation_loader = DataLoader(validation_dataset, num_workers=h.num_workers, shuffle=False,
                                   sampler=None,
                                   batch_size=1,
                                   pin_memory=True,
                                   drop_last=True)

    generator = get_module_from_config(
        get_static_generator_config(
            initial_skip_ratio=h.initial_skip_ratio,
            expansion_size=h.gen_expansion_size
        )
    )
    for p in generator.parameters():
        p.data.fill_(1e-8)
    summary(generator,
            input_size=(1, h.segment_size),
            batch_size=h.batch_size,
            device='cpu')

    discriminator = get_module_from_config(
        get_static_all_in_one_discriminator(
            expansion_size=h.disc_expansion_size
        )
    )
    summary(discriminator,
            input_size=(1, h.segment_size),
            batch_size=h.batch_size,
            device='cpu')

    example_item = train_dataset.label_option_groups
    keepers = generate_keepers_by_example(h.embedding_size, example_item)
    hunters = generate_hunters_by_example(h.embedding_size, example_item)

    for key, keeper in keepers.items():
        print(f'{key} keeper:')
        summary(keeper,
                input_size=(h.embedding_size, h.segment_size // h.embedding_size),
                batch_size=h.batch_size,
                device='cpu')

    for key, hunter in hunters.items():
        print(f'{key} hunter:')
        summary(hunter,
                input_size=(h.embedding_size, h.segment_size // h.embedding_size),
                batch_size=h.batch_size,
                device='cpu')

    # model
    model = GanAutoencoder(generator, discriminator, keepers, hunters, h, a)
    # model = LitModel(generator, 0.0002, 0.8, 0.99)

    experiment_name = h.experiment_name

    # training
    trainer = pl.Trainer(
        gpus=1,
        num_nodes=1,
        precision=32,
        # limit_train_batches=0.5,
        max_steps=10000000,
        logger=pl_loggers.TensorBoardLogger(
            '/mount/logs/',
            name=experiment_name,
            version=h.version,
            default_hp_metric=False),
        val_check_interval=500,
        num_sanity_val_steps=h.visualizations_amount,
        callbacks=[
            ContinuousCheckpointCallback(100),
            HistoryCheckpointCallback(5000),
            OutputLoggingCallback({
                'train': 20,
                'validation': 500
            }),
            ManualOptimizationCallback(h.accumulated_grad_batches),
            ValveDecayCallback(
                valves_config=h.valves,
                valves_steps=h.valves_steps
            ),
            GanValidationVisualizationCallback(h.visualizations_amount),
            GanModelsGraphVisualizationCallback(),
            GlobalSyncCallback()
        ]
    )
    trainer.fit(model, train_loader, validation_loader)
    # result = trainer.tune(model, train_loader, validation_loader)
    # print('best lr: {}'.format(result['lr_find'].suggestion()))
    # trainer.fit(model, sanity_loader, sanity_val_loader)
    # trainer.fit(model, train_loader, validation_loader)


if __name__ == '__main__':
    main()
