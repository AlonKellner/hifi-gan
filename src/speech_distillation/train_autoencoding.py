import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
import os
import time
import argparse
import json
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DistributedSampler, DataLoader
import torch.multiprocessing as mp
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel
from src.env import AttrDict, build_env
from src.meldataset import mel_spectrogram, get_dataset_filelist
from src.utils import plot_spectrogram, scan_checkpoint, load_checkpoint, save_checkpoint
import math

from torchsummary import summary

from static_configs import get_static_generator_config, \
    get_static_all_in_one_discriminator
from datasets import WaveDataset
from configurable_module import get_module_from_config
from custom_losses import recursive_loss, plus_mean_loss, minus_mean_loss
from custom_blocks import get_modules, ValveBlock

torch.backends.cudnn.benchmark = True


def train(rank, a, h):
    if h.num_gpus > 1:
        init_process_group(backend=h.dist_config['dist_backend'], init_method=h.dist_config['dist_url'],
                           world_size=h.dist_config['world_size'] * h.num_gpus, rank=rank)

    torch.cuda.manual_seed(h.seed)
    device = torch.device('cuda:{:d}'.format(rank))

    generator = get_module_from_config(get_static_generator_config()).to(device)
    summary(generator,
            input_size=(1, h.segment_size),
            batch_size=h.batch_size)

    valves_config = h.valves
    valves_modules = {}
    for valve_tag, valve_config in valves_config.items():
        anti_valve_tag = valve_config['anti']
        valve_modules = get_modules(generator, ValveBlock, [valve_tag])
        anti_valve_modules = get_modules(generator, ValveBlock, [anti_valve_tag])
        valves_modules[valve_tag] = (valve_modules, anti_valve_modules)

    discriminator = get_module_from_config(get_static_all_in_one_discriminator(8)).to(device)
    summary(discriminator,
            input_size=(1, h.segment_size),
            batch_size=h.batch_size)

    if rank == 0:
        os.makedirs(a.checkpoint_path, exist_ok=True)
        print("checkpoints directory : ", a.checkpoint_path)

    if os.path.isdir(a.checkpoint_path):
        cp_g = scan_checkpoint(a.checkpoint_path, 'g_')
        cp_do = scan_checkpoint(a.checkpoint_path, 'do_')

    steps = 0
    if cp_g is None or cp_do is None:
        state_dict_do = None
        last_epoch = -1
    else:
        state_dict_g = load_checkpoint(cp_g, device)
        state_dict_do = load_checkpoint(cp_do, device)
        generator.load_state_dict(state_dict_g['generator'])
        discriminator.load_state_dict(state_dict_do['discriminator'])
        steps = state_dict_do['steps'] + 1
        last_epoch = state_dict_do['epoch']

    if h.num_gpus > 1:
        generator = DistributedDataParallel(generator, device_ids=[rank]).to(device)
        discriminator = DistributedDataParallel(discriminator, device_ids=[rank]).to(device)

    accumulation_steps = h.accumulated_grad_batches

    optim_g = torch.optim.AdamW(
        generator.parameters(),
        h.gen_learning_rate,
        betas=(h.gen_adam_b1, h.gen_adam_b2)
    )
    optim_d = torch.optim.AdamW(
        discriminator.parameters(),
        h.disc_learning_rate,
        betas=(h.disc_adam_b1, h.disc_adam_b2)
    )

    if state_dict_do is not None:
        optim_g.load_state_dict(state_dict_do['optim_g'])
        optim_d.load_state_dict(state_dict_do['optim_d'])

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=h.lr_decay, last_epoch=last_epoch)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=h.lr_decay, last_epoch=last_epoch)

    training_filelist, validation_filelist = get_dataset_filelist(a)

    trainset = WaveDataset(training_filelist, h.segment_size, h.sampling_rate, False, n_cache_reuse=0,
                           fine_tuning=a.fine_tuning)

    train_sampler = DistributedSampler(trainset) if h.num_gpus > 1 else None

    train_loader = DataLoader(trainset, num_workers=h.num_workers, shuffle=False,
                              sampler=train_sampler,
                              batch_size=h.batch_size,
                              pin_memory=True,
                              drop_last=True)

    if rank == 0:
        validset = WaveDataset(validation_filelist, h.validation_segment_size, h.sampling_rate, False, n_cache_reuse=0,
                               fine_tuning=a.fine_tuning, deterministic=True)
        validation_loader = DataLoader(validset, num_workers=1, shuffle=False,
                                       sampler=None,
                                       batch_size=1,
                                       pin_memory=True,
                                       drop_last=True)

        sw = SummaryWriter(os.path.join(a.checkpoint_path, 'logs'))

    generator.train()
    discriminator.train()
    for epoch in range(max(0, last_epoch), a.training_epochs):
        if rank == 0:
            start = time.time()
            print("Epoch: {}".format(epoch + 1))

        if h.num_gpus > 1:
            train_sampler.set_epoch(epoch)

        for i, batch in enumerate(train_loader):
            should_backprop_reconstruction = True
            should_backprop_discrimination = True
            should_backprop_adversarial = True
            should_step_discriminator = steps % accumulation_steps == 0
            should_step_generator = steps % accumulation_steps == 0
            should_step_valves = steps % h.valves_steps == 0

            if rank == 0:
                start_b = time.time()
            y, _ = batch
            y = torch.autograd.Variable(y.to(device, non_blocking=True))
            y = y.unsqueeze(1)

            y_generated = generator(y)
            gen_current_loss = 0
            disc_current_loss = 0

            if should_backprop_reconstruction:
                reconstruction_loss, mel_loss, wave_loss, _, _ = get_reconstruction_loss(y, y_generated, h)
                gen_current_loss = reconstruction_loss + gen_current_loss
            if should_backprop_adversarial:
                adversarial_loss, disc_loss, fmap_loss = get_adversarial_loss(y, y_generated, discriminator)
                gen_current_loss = adversarial_loss + gen_current_loss
            if gen_current_loss != 0:
                gen_current_loss.backward()

            if should_backprop_discrimination:
                discrimination_loss, real_loss, fake_loss = get_discrimination_loss(y, y_generated, discriminator)
                disc_current_loss = discrimination_loss + disc_current_loss
            if disc_current_loss != 0:
                disc_current_loss.backward()

            if should_step_generator:
                optim_g.step()
                optim_g.zero_grad()
            if should_step_discriminator:
                optim_d.step()
                optim_d.zero_grad()

            if rank == 0:
                # STDOUT logging
                if steps % a.stdout_interval == 0:
                    message_lines = [
                        'Steps : {:d}'.format(steps),
                        'Gen Total Error : {:4.3f}'.format(gen_current_loss)
                    ]
                    with torch.no_grad():
                        if should_backprop_reconstruction:
                            mel_error = mel_loss.item()
                            wave_error = wave_loss.item()
                            message_lines.append('Gen Mel Error : {:4.3f}'.format(mel_error))
                            message_lines.append('Gen Wave Error : {:4.3f}'.format(wave_error))
                        if should_backprop_discrimination:
                            real_error = real_loss
                            fake_error = fake_loss
                            message_lines.append('Disc Real Error : {:4.3f}'.format(fake_error))
                            message_lines.append('Disc Fake Error : {:4.3f}'.format(real_error))
                        if should_backprop_adversarial:
                            disc_error = disc_loss.item()
                            fmap_error = fmap_loss.item()
                            message_lines.append('Gen Disc Error : {:4.3f}'.format(disc_error))
                            message_lines.append('Gen FMap Error : {:4.3f}'.format(fmap_error))
                    message_lines.append('batch seconds : {:4.3f}'.format(time.time() - start_b))
                    print(', '.join(message_lines))

                # checkpointing
                if steps % a.checkpoint_interval == 0:  # and steps != 0:
                    checkpoint_path = "{}/g_{:08d}".format(a.checkpoint_path, steps)
                    save_checkpoint(checkpoint_path,
                                    {'generator': (generator.module if h.num_gpus > 1 else generator).state_dict()})
                    checkpoint_path = "{}/do_{:08d}".format(a.checkpoint_path, steps)
                    save_checkpoint(checkpoint_path,
                                    {
                                        'discriminator': (discriminator.module if h.num_gpus > 1
                                                          else discriminator).state_dict(),
                                        'optim_g': optim_g.state_dict(),
                                        'optim_d': optim_d.state_dict(),
                                        'steps': steps,
                                        'epoch': epoch
                                    })

                # Tensorboard summary logging
                if steps % a.summary_interval == 0:
                    sw.add_scalar("training_generator/total", gen_current_loss, steps)
                    sw.add_scalar("training_discriminator/total", disc_current_loss, steps)
                    if should_backprop_reconstruction:
                        sw.add_scalar("training_generator/reconstruction/mel", mel_error, steps)
                        sw.add_scalar("training_generator/reconstruction/wave", wave_error, steps)
                    if should_backprop_discrimination:
                        sw.add_scalar("training_discriminator/real", real_error, steps)
                        sw.add_scalar("training_discriminator/fake", fake_error, steps)
                    if should_backprop_adversarial:
                        sw.add_scalar("training_generator/adversarial/disc", disc_error, steps)
                        sw.add_scalar("training_generator/adversarial/fmap", fmap_error, steps)

                # Validation
                if steps % a.validation_interval == 0: # and steps != 0:
                    generator.eval()
                    torch.cuda.empty_cache()
                    wave_err_tot = 0
                    mel_err_tot = 0
                    with torch.no_grad():
                        for j, batch in enumerate(validation_loader):
                            y, _ = batch
                            y = y.to(device)
                            y_generated = generator(y.unsqueeze(1))
                            y_diff = y - y_generated
                            reconstruction_loss, mel_loss, wave_loss, y_mel, y_generated_mel = \
                                get_reconstruction_loss(y, y_generated, h)
                            y_diff_mel = mel_spectrogram(y_diff.squeeze(1), h.n_fft, h.num_mels, h.sampling_rate,
                                                         h.hop_size,
                                                         h.win_size,
                                                         h.fmin, h.fmax_for_loss)

                            y_mel_diff_inverse = y_mel - y_generated_mel
                            wave_err_tot = wave_loss + wave_err_tot
                            mel_err_tot = mel_loss + mel_err_tot

                            if j <= 4:
                                if steps == 0:
                                    sw.add_audio('ground_truth/y_{}'.format(j), y[0], steps, h.sampling_rate)
                                    sw.add_figure('ground_truth/y_mel_{}'.format(j),
                                                  plot_spectrogram(y_mel[0].cpu().numpy()),
                                                  steps)

                                sw.add_audio('generated/y_{}'.format(j), y_generated[0], steps,
                                             h.sampling_rate)
                                sw.add_figure('generated/y_mel_{}'.format(j),
                                              plot_spectrogram(y_generated_mel.squeeze(0).cpu().numpy()), steps)

                                sw.add_audio('wave_diff/y_{}'.format(j), y_diff[0], steps,
                                             h.sampling_rate)
                                sw.add_figure('wave_diff/y_mel_{}'.format(j),
                                              plot_spectrogram(y_diff_mel.squeeze(0).cpu().numpy()), steps)
                                sw.add_figure('mel_diff/y_mel_{}'.format(j),
                                              plot_spectrogram(y_mel_diff_inverse.squeeze(0).cpu().numpy()), steps)

                        wave_err = wave_err_tot / (j + 1)
                        sw.add_scalar("validation/wave", wave_err, steps)
                        mel_err = mel_err_tot / (j + 1)
                        sw.add_scalar("validation/mel", mel_err, steps)

                    generator.train()

            if should_step_valves:
                for valve_tag, all_valve_modules in valves_modules.items():
                    valve_modules, anti_valves_modules = all_valve_modules
                    valve_config = valves_config[valve_tag]
                    valve_limit = valve_config['limit']
                    if valve_limit < steps:
                        pow_decay = 0
                        anti_pow_decay = 0
                    else:
                        valve_decay = valve_config['decay']
                        pow_decay = math.pow(valve_decay, h.valves_steps)

                        anti_valve_decay = valve_config['anti_decay']
                        anti_pow_decay = math.pow(anti_valve_decay, h.valves_steps)
                    for valve_module in valve_modules:
                        valve_module.ratio *= pow_decay
                    for anti_valve_module in anti_valves_modules:
                        anti_valve_module.ratio = (1 - (1 - anti_valve_module.ratio) * anti_pow_decay)

            steps += 1

        scheduler_g.step()
        scheduler_d.step()

        if rank == 0:
            print('Time taken for epoch {} is {} sec\n'.format(epoch + 1, int(time.time() - start)))


def get_reconstruction_loss(y, y_generated, h):
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


def get_discrimination_loss(y, y_generated, discriminator):
    y_mom_r, y_fmap_r = discriminator(y)
    y_all_r, y_all_var_r = y_mom_r
    y_d_r, y_sub_d_r = y_all_r
    y_mom_g_detach, y_fmap_g_detach = discriminator(y_generated.detach())
    y_all_g_detach, y_all_var_g_detach = y_mom_g_detach
    y_d_g_detach, y_sub_d_g_detach = y_all_g_detach

    loss_disc_r = plus_mean_loss(y_d_r)
    loss_sub_disc_r = sum(plus_mean_loss(sub_d) for sub_d in y_sub_d_r)
    loss_all_disc_r = loss_disc_r + loss_sub_disc_r

    loss_disc_g = minus_mean_loss(y_d_g_detach)
    loss_sub_disc_g = sum(minus_mean_loss(sub_d) for sub_d in y_sub_d_g_detach)
    loss_all_disc_g = loss_disc_g + loss_sub_disc_g

    loss_all_disc = loss_all_disc_r + loss_all_disc_g
    return loss_all_disc, loss_all_disc_r, loss_all_disc_g


def get_adversarial_loss(y, y_generated, discriminator):
    y_mom_r, y_fmap_r = discriminator(y)
    y_mom_g, y_fmap_g = discriminator(y_generated)
    y_all_g, y_all_var_g = y_mom_g
    y_d_g, y_sub_d_g = y_all_g

    loss_disc = plus_mean_loss(y_d_g)
    loss_sub_disc = sum(plus_mean_loss(sub_d) for sub_d in y_sub_d_g)
    loss_all_disc = (loss_disc + loss_sub_disc) * 0.003

    loss_fm = recursive_loss(y_fmap_r, y_fmap_g) * 0.003

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

    torch.manual_seed(h.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        h.num_gpus = torch.cuda.device_count()
        h.batch_size = int(h.batch_size / h.num_gpus)
        print('Batch size per GPU :', h.batch_size)
    else:
        pass

    if h.num_gpus > 1:
        mp.spawn(train, nprocs=h.num_gpus, args=(a, h,))
    else:
        train(0, a, h)


if __name__ == '__main__':
    main()
