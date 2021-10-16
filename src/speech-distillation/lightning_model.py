from torch.nn import functional as F
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
import os
import argparse
import json
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DistributedSampler, DataLoader
from src.env import AttrDict, build_env
from src.meldataset import mel_spectrogram, get_dataset_filelist
from src.utils import plot_spectrogram
import math

from torchsummary import summary

from static_configs import get_static_generator_config, \
    get_static_all_in_one_discriminator
from datasets import WaveDataset
from configurable_module import get_module_from_config
from custom_losses import feature_loss, one_loss, zero_loss
from custom_blocks import get_modules, ValveBlock
from lit_model import LitModel

torch.backends.cudnn.benchmark = True


class GanAutoencoder(pl.LightningModule):
    def __init__(self, generator, discriminator, config, args, train_g=True, train_d=True):
        super().__init__()
        self.save_hyperparameters(config)
        self.save_hyperparameters(args)
        self.automatic_optimization = not (train_g and train_d)
        self.train_g = train_g
        self.train_d = train_d
        self.accumulated_grad_batches = config.accumulated_grad_batches
        self.gen_learning_rate = config.gen_learning_rate
        self.gen_adam_b1 = config.gen_adam_b1
        self.gen_adam_b2 = config.gen_adam_b2
        self.disc_learning_rate = config.disc_learning_rate
        self.disc_adam_b1 = config.disc_adam_b1
        self.disc_adam_b2 = config.disc_adam_b2
        self.generator = generator
        self.discriminator = discriminator
        self.config = config
        self.args = args

        self.valves_config = self.config.valves
        self.valves_modules = {}
        for valve_tag, valve_config in self.valves_config.items():
            anti_valve_tag = valve_config['anti']
            valve_modules = get_modules(generator, ValveBlock, [valve_tag])
            anti_valve_modules = get_modules(generator, ValveBlock, [anti_valve_tag])
            self.valves_modules[valve_tag] = (valve_modules, anti_valve_modules)

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
        optimizers = []
        schedulers = []
        if self.train_g:
            self.optim_g = optim_g
            optimizers.append(optim_g)
            schedulers.append(scheduler_g)
        if self.train_d:
            self.optim_d = optim_d
            optimizers.append(optim_d)
            schedulers.append(scheduler_d)
        return optimizers, schedulers

    def training_step(self, train_batch, batch_idx):
        should_backprop_reconstruction = True
        should_backprop_discrimination = True
        should_backprop_adversarial = True
        should_step_valves = self.global_step % self.config.valves_steps == 0

        should_step_discriminator = self.train_d and self.global_step % self.accumulated_grad_batches == 0
        should_step_generator = self.train_g and self.global_step % self.accumulated_grad_batches == 0

        y, _ = train_batch
        y = y.unsqueeze(1)

        y_generated = self.generator(y)
        gen_current_loss = 0
        disc_current_loss = 0

        if should_backprop_reconstruction:
            reconstruction_loss, mel_loss, wave_loss, _, _ = self.get_reconstruction_loss(y, y_generated, self.config)
            gen_current_loss = reconstruction_loss + gen_current_loss
        if should_backprop_adversarial:
            adversarial_loss, disc_loss, fmap_loss = self.get_adversarial_loss(y, y_generated, self.discriminator)
            gen_current_loss = adversarial_loss + gen_current_loss

        if should_backprop_discrimination:
            discrimination_loss, real_loss, fake_loss = self.get_discrimination_loss(y, y_generated, self.discriminator)
            disc_current_loss = discrimination_loss + disc_current_loss

        if not self.automatic_optimization:
            self.manual_backward(gen_current_loss)
            self.manual_backward(disc_current_loss)

            if should_step_generator:
                self.optim_g.step()
                self.optim_g.zero_grad()
            if should_step_discriminator:
                self.optim_d.step()
                self.optim_d.zero_grad()

        losses_dict = {
            'training_generator/total': gen_current_loss,
            'training_discriminator/total': disc_current_loss,
        }

        sw = self.logger.experiment
        if self.global_step % self.args.summary_interval == 0:
            sw.add_scalar('training_generator/total', gen_current_loss, self.global_step)
            sw.add_scalar('training_discriminator/total', disc_current_loss, self.global_step)
            if should_backprop_reconstruction:
                losses_dict['training_generator/reconstruction/mel'] = mel_loss
                losses_dict['training_generator/reconstruction/wave'] = wave_loss
                sw.add_scalar('training_generator/reconstruction/mel', mel_loss, self.global_step)
                sw.add_scalar('training_generator/reconstruction/wave', wave_loss, self.global_step)
            if should_backprop_discrimination:
                losses_dict['training_discriminator/real'] = real_loss
                losses_dict['training_discriminator/fake'] = fake_loss
                sw.add_scalar('training_discriminator/real', real_loss, self.global_step)
                sw.add_scalar('training_discriminator/fake', fake_loss, self.global_step)
            if should_backprop_adversarial:
                losses_dict['training_generator/adversarial/disc'] = disc_loss
                losses_dict['training_generator/adversarial/fmap'] = fmap_loss
                sw.add_scalar('training_generator/adversarial/disc', disc_loss, self.global_step)
                sw.add_scalar('training_generator/adversarial/fmap', fmap_loss, self.global_step)

        if should_step_valves:
            for valve_tag, all_valve_modules in self.valves_modules.items():
                valve_modules, anti_valves_modules = all_valve_modules
                valve_config = self.valves_config[valve_tag]
                valve_limit = valve_config['limit']
                if valve_limit < self.global_step:
                    pow_decay = 0
                    anti_pow_decay = 0
                else:
                    valve_decay = valve_config['decay']
                    pow_decay = math.pow(valve_decay, self.config.valves_steps)

                    anti_valve_decay = valve_config['anti_decay']
                    anti_pow_decay = math.pow(anti_valve_decay, self.config.valves_steps)
                for valve_module in valve_modules:
                    valve_module.ratio *= pow_decay
                for anti_valve_module in anti_valves_modules:
                    anti_valve_module.ratio = (1 - (1 - anti_valve_module.ratio) * anti_pow_decay)

        if not self.train_g:
            return disc_current_loss
        if not self.train_d:
            return gen_current_loss

        schedulers = self.lr_schedulers()
        for scheduler in schedulers:
            scheduler.step()

        return losses_dict

    def validation_step(self, val_batch, batch_idx):
        y, _ = val_batch
        y_generated = self.generator(y.unsqueeze(1))
        y_diff = y - y_generated
        reconstruction_loss, mel_loss, wave_loss, y_mel, y_generated_mel = \
            self.get_reconstruction_loss(y, y_generated, self.config)
        y_diff_mel = mel_spectrogram(
            y_diff.squeeze(1),
            self.config.n_fft,
            self.config.num_mels,
            self.config.sampling_rate,
            self.config.hop_size,
            self.config.win_size,
            self.config.fmin,
            self.config.fmax_for_loss)

        y_mel_diff_inverse = y_mel - y_generated_mel

        sw = self.logger.experiment
        if batch_idx < 5:
            if self.global_step == 0:
                sw.add_audio('ground_truth/y_{}'.format(batch_idx), y[0], self.global_step, self.config.sampling_rate)
                sw.add_figure('ground_truth/y_mel_{}'.format(batch_idx),
                              plot_spectrogram(y_mel[0].cpu().numpy()),
                              self.global_step)

            sw.add_audio('generated/y_{}'.format(batch_idx), y_generated[0], self.global_step,
                         self.config.sampling_rate)
            sw.add_figure('generated/y_mel_{}'.format(batch_idx),
                          plot_spectrogram(y_generated_mel.squeeze(0).cpu().numpy()), self.global_step)

            sw.add_audio('wave_diff/y_{}'.format(batch_idx), y_diff[0], self.global_step,
                         self.config.sampling_rate)
            sw.add_figure('wave_diff/y_mel_{}'.format(batch_idx),
                          plot_spectrogram(y_diff_mel.squeeze(0).cpu().numpy()), self.global_step)
            sw.add_figure('mel_diff/y_mel_{}'.format(batch_idx),
                          plot_spectrogram(y_mel_diff_inverse.squeeze(0).cpu().numpy()), self.global_step)
            return {
                'validation/wave': wave_loss,
                'validation/mel': mel_loss
            }

    def get_reconstruction_loss(self, y, y_generated, h):
        y_mel = mel_spectrogram(y.squeeze(1), h.n_fft, h.num_mels, h.sampling_rate, h.hop_size,
                                h.win_size,
                                h.fmin, h.fmax_for_loss)
        y_generated_mel = mel_spectrogram(y_generated.squeeze(1), h.n_fft, h.num_mels, h.sampling_rate, h.hop_size,
                                          h.win_size,
                                          h.fmin, h.fmax_for_loss)
        loss_mel = F.l1_loss(y_mel, y_generated_mel) * 10
        loss_wave = F.l1_loss(y, y_generated) * 350
        loss_recon = (loss_mel + loss_wave)
        return loss_recon, loss_mel, loss_wave, y_mel, y_generated_mel

    def get_discrimination_loss(self, y, y_generated, discriminator):
        y_mom_r, y_fmap_r = discriminator(y)
        y_all_r, y_all_var_r = y_mom_r
        y_d_r, y_sub_d_r = y_all_r
        y_mom_g_detach, y_fmap_g_detach = discriminator(y_generated.detach())
        y_all_g_detach, y_all_var_g_detach = y_mom_g_detach
        y_d_g_detach, y_sub_d_g_detach = y_all_g_detach

        loss_disc_r = one_loss(y_d_r)
        loss_sub_disc_r = sum(one_loss(sub_d) for sub_d in y_sub_d_r)
        loss_all_disc_r = loss_disc_r + loss_sub_disc_r

        loss_disc_g = zero_loss(y_d_g_detach)
        loss_sub_disc_g = sum(zero_loss(sub_d) for sub_d in y_sub_d_g_detach)
        loss_all_disc_g = loss_disc_g + loss_sub_disc_g

        loss_all_disc = loss_all_disc_r + loss_all_disc_g
        return loss_all_disc, loss_all_disc_r, loss_all_disc_g

    def get_adversarial_loss(self, y, y_generated, discriminator):
        y_mom_r, y_fmap_r = discriminator(y)
        y_mom_g, y_fmap_g = discriminator(y_generated)
        y_all_g, y_all_var_g = y_mom_g
        y_d_g, y_sub_d_g = y_all_g

        loss_disc = one_loss(y_d_g)
        loss_sub_disc = sum(one_loss(sub_d) for sub_d in y_sub_d_g)
        loss_all_disc = (loss_disc + loss_sub_disc) * 0.01

        loss_fm = feature_loss(y_fmap_r, y_fmap_g) * 0.1

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

    training_filelist, validation_filelist = get_dataset_filelist(a)

    sanityset = WaveDataset(training_filelist[0:10], h.segment_size, h.sampling_rate, False, n_cache_reuse=0,
                            fine_tuning=a.fine_tuning)

    sanity_sampler = DistributedSampler(sanityset) if h.num_gpus > 1 else None

    sanity_loader = DataLoader(sanityset, num_workers=h.num_workers, shuffle=False,
                               sampler=sanity_sampler,
                               batch_size=h.batch_size,
                               pin_memory=True,
                               drop_last=True)

    sanityvalset = WaveDataset(validation_filelist[0:10], h.validation_segment_size, h.sampling_rate, False,
                               n_cache_reuse=0,
                               fine_tuning=a.fine_tuning, deterministic=True)
    sanity_val_loader = DataLoader(sanityvalset, num_workers=1, shuffle=False,
                                   sampler=None,
                                   batch_size=1,
                                   pin_memory=True,
                                   drop_last=True)

    trainset = WaveDataset(training_filelist, h.segment_size, h.sampling_rate, False, n_cache_reuse=0,
                           fine_tuning=a.fine_tuning)

    train_sampler = DistributedSampler(trainset) if h.num_gpus > 1 else None

    train_loader = DataLoader(trainset, num_workers=h.num_workers, shuffle=False,
                              sampler=train_sampler,
                              batch_size=h.batch_size,
                              pin_memory=True,
                              drop_last=True)

    validset = WaveDataset(validation_filelist, h.validation_segment_size, h.sampling_rate, False, n_cache_reuse=0,
                           fine_tuning=a.fine_tuning, deterministic=True)
    validation_loader = DataLoader(validset, num_workers=1, shuffle=False,
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

    # model
    model = GanAutoencoder(generator, discriminator, h, a)
    # model = LitModel(generator, 0.0002, 0.8, 0.99)

    # training
    trainer = pl.Trainer(
        gpus=1,
        num_nodes=1,
        precision=32,
        limit_train_batches=0.5,
        max_steps=1000000,
        # auto_lr_find='learning_rate'
        auto_lr_find='gen_learning_rate',
        logger=pl_loggers.TensorBoardLogger('/mount/logs/', name=h.experiment_name),
        val_check_interval=500,
        callbacks=[

        ]
    )
    trainer.fit(model, train_loader, validation_loader)
    # result = trainer.tune(model, train_loader, validation_loader)
    # print('best lr: {}'.format(result['lr_find'].suggestion()))
    # trainer.fit(model, sanity_loader, sanity_val_loader)
    # trainer.fit(model, train_loader, validation_loader)


if __name__ == '__main__':
    main()
