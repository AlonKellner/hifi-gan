import os
import warnings
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

from src.speech_distillation.best_checkpoint_callback import BestCheckpointCallback
from src.speech_distillation.global_sync_callback import GlobalSyncCallback
from src.speech_distillation.global_sync_lr_scheduler import GlobalSyncDecoratorLR, GlobalSyncExponentialLR
from src.speech_distillation.label_bias_sniffer import generate_sniffers_by_example, LabelBiasSniffer
from src.speech_distillation.output_sum_callback import OutputSumCallback
from src.speech_distillation.yaml_utils import do_and_cache

warnings.simplefilter(action='ignore', category=FutureWarning)
import argparse
import json
import torch
from torch.utils.data import DistributedSampler, DataLoader
from src.env import AttrDict
from multilabel_wave_dataset import MultilabelWaveDataset
from src.meldataset import mel_spectrogram
from src.speech_distillation.continuous_checkpoint_callback import ContinuousCheckpointCallback
from src.speech_distillation.history_checkpoint_callback import HistoryCheckpointCallback
from src.speech_distillation.output_logging_callback import OutputLoggingCallback
from src.speech_distillation.manual_optimization_callback import ManualOptimizationCallback
from src.speech_distillation.valve_decay_callback import ValveDecayCallback
from src.speech_distillation.gan_validation_visualization_callback import \
    GanValidationVisualizationCallback

from torchsummary import summary, InputSize, RandInt

from static_configs import get_static_generator_config, \
    get_static_all_in_one_discriminator
from configurable_module import get_module_from_config
from custom_losses import recursive_loss, get_losses_by_types

torch.backends.cudnn.benchmark = True

from embedding_classifiers.embedding_classifiers_static_configs import generate_keepers_by_example, \
    generate_hunters_by_example


class GanAutoencoder(pl.LightningModule):
    def __init__(self,
                 generator,
                 discriminator, discriminator_copy,
                 keepers,
                 hunters, hunters_copies,
                 sniffers,
                 config, args, config_dict):
        super().__init__()
        self.automatic_optimization = False
        self.accumulated_grad_batches = config.accumulated_grad_batches

        self.learning_rates = config_dict['learning_rates']
        self.loss_factors = config_dict['loss_factors']
        self.losses = get_losses_by_types(config_dict['loss_funcs'])

        self.generator = generator
        self.discriminator = discriminator
        self.discriminator_copy = discriminator_copy
        self.keepers = keepers
        self.hunters = hunters
        self.hunters_copies = hunters_copies
        self.sniffers = sniffers
        self.learning_models = {
            'generator': self.generator,
            'discriminator': self.discriminator,
            'keepers': self.keepers,
            'hunters': self.hunters,
            'sniffers': self.sniffers
        }
        self.flat_learning_models = self._create_flat_models(self.learning_models)

        self.config = config
        self.args = args

        self._update_discriminator_copy()
        self._update_hunters_copies()

    def _update_discriminator_copy(self):
        self.discriminator_copy.load_state_dict(self.discriminator.state_dict())

    def _update_hunters_copies(self):
        for key, hunter in self.hunters.items():
            self.hunters_copies[key].load_state_dict(hunter.state_dict())

    def _create_flat_models(self, models):
        if isinstance(models, (dict, torch.nn.ModuleDict)):
            models_2d_array = {key: self._create_flat_models(model) for key, model in
                               models.items()}
            flat_models = {
                (f'{key_2d}' if key_1d is None else f'{key_2d}/{key_1d}'): model
                for key_2d, models_1d_array in models_2d_array.items()
                for key_1d, model in models_1d_array.items()
            }
            return flat_models
        else:
            return {None: models}

    def get_learning_models(self):
        return self.flat_learning_models

    def forward(self, x):
        return self.generator(x)

    def configure_optimizers(self):
        optimizers = self._models_to_optimizers(self.learning_models, self.learning_rates)

        schedulers = [
            GlobalSyncExponentialLR(optimizer, lr_decay=self.config.lr_decay, global_step=self.global_step) for
            optimizer in optimizers
        ]
        return optimizers, schedulers

    def _models_to_optimizers(self, models, learning_rates):
        if isinstance(models, (dict, torch.nn.ModuleDict)):
            optimizers_2d_array = [self._models_to_optimizers(model, learning_rates[key]) for key, model in
                                   models.items()]
            flat_optimizers = [
                optimizer
                for optimizers_1d_array in optimizers_2d_array
                for optimizer in optimizers_1d_array
            ]
            return flat_optimizers
        else:
            return [torch.optim.AdamW(
                models.parameters(),
                learning_rates,
                betas=(self.config.adam_b1, self.config.adam_b2),
                amsgrad=True
            )]

    def training_step(self, train_batch, batch_idx):
        losses_dict = self.get_losses(train_batch, batch_idx, backprop=True)
        self._update_discriminator_copy()
        self._update_hunters_copies()
        return losses_dict

    def validation_step(self, val_batch, batch_idx):
        losses_dict = self.get_losses(val_batch, batch_idx, backprop=False)
        return losses_dict

    def get_losses(self, batch, batch_idx, backprop=True):
        wav, wav_path, time_labels, labels = batch
        wav = wav.unsqueeze(1)

        wav_generated, embeddings = self.generator(wav)
        (embeddings,) = embeddings

        reconstruction_data = self.get_reconstruction_data(wav_generated, wav)
        adversarial_discriminator_data = self.get_adversarial_data(wav_generated, wav, self.discriminator_copy)
        keepers_data = {'keepers': self.get_embeddings_data(embeddings, time_labels, self.keepers)}
        adversarial_hunters_data = self.get_adversarial_hunting_data(embeddings, time_labels, self.hunters_copies,
                                                                     self.keepers,
                                                                     self.sniffers)
        generator_data = self._merge_dicts(reconstruction_data, adversarial_discriminator_data, keepers_data,
                                           adversarial_hunters_data)

        detached_embeddings = self._detach_recursively(embeddings)
        detached_wav_generated = self._detach_recursively(wav_generated)

        discriminator_data = self.get_discriminator_data(detached_wav_generated, wav,
                                                         self.discriminator)

        hunters_data = self.get_embeddings_data(detached_embeddings, time_labels,
                                                self.hunters)

        sniffers_data = self.get_sniffers_data(detached_embeddings, time_labels, self.keepers,
                                               self.sniffers)

        all_data = {
            'generator': generator_data,
            'discriminator': discriminator_data,
            'hunters': hunters_data,
            'sniffers': sniffers_data
        }
        all_losses = {}
        for key, data in all_data.items():
            current_losses_dict, current_losses_sum = self._calculate_losses(self.losses[key], self.loss_factors[key],
                                                                             data, backprop=backprop)
            all_losses[key] = current_losses_dict
        return all_losses

    def _calculate_losses(self, loss, factor, data, backprop=True):
        if isinstance(data, tuple) \
                and len(data) == 2 \
                and callable(data[0]) \
                and isinstance(data[1], tuple):
            data = data[0](*data[1])
        if isinstance(loss, dict):
            losses_sum = 0
            losses = {}
            for key in loss.keys():
                current_losses, current_sum = self._calculate_losses(loss[key], factor[key], data[key],
                                                                     backprop=backprop)
                losses[key] = current_losses
                losses_sum = current_sum + losses_sum
            losses['total'] = losses_sum
            return losses, losses_sum
        else:
            current_loss = recursive_loss(loss, *data) * factor
            if backprop:
                self.manual_backward(current_loss, retain_graph=True)
            current_loss = current_loss.detach()
            loss_sum = current_loss
            return current_loss, loss_sum

    def _detach_recursively(self, losses):
        if isinstance(losses, dict):
            return {key: self._detach_recursively(loss) for key, loss in losses.items()}
        elif isinstance(losses, torch.Tensor):
            return losses.detach()
        else:
            return losses

    def _merge_dicts(self, *all_dicts):
        result_dict = {}
        for current_dict in all_dicts:
            result_dict = self._merge_into(result_dict, current_dict)
        return result_dict

    def _merge_into(self, base, remote):
        if isinstance(remote, dict):
            for key, remote_value in remote.items():
                if key in base:
                    base[key] = self._merge_into(base[key], remote_value)
                else:
                    base[key] = remote_value
        return base

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

    def get_adversarial_hunting_data(self, embeddings, time_labels, hunters, keepers, sniffers):
        def get(_embedding, _time_labels, _hunter, _keeper, _sniffer, _key):
            with torch.no_grad():
                time_labels_keeper_predictions = _keeper(_embedding)[0]
                bias_labels = _sniffer(time_labels_keeper_predictions[_key])[0]
            time_labels_predictions = _hunter(_embedding)[0]
            return time_labels_predictions, bias_labels, _time_labels

        hunt_data = {}
        for key, hunter in hunters.items():
            keeper = keepers[key]
            sniffer = sniffers[key]
            embedding = embeddings[key]
            hunt_data[key] = get, (embedding, time_labels, hunter, keeper, sniffer, key)

        data = {
            'adversarial': {
                'hunters': hunt_data
            }
        }
        return data

    def get_sniffers_data(self, embeddings, time_labels, keepers, sniffers):
        def get(_embedding, _time_labels, _keeper, _sniffer, _key):
            with torch.no_grad():
                time_labels_keeper_predictions = _keeper(_embedding)[0]
            bias_labels = _sniffer(time_labels_keeper_predictions[_key])[0]
            return bias_labels, _time_labels

        sniff_data = {}
        for key, sniffer in sniffers.items():
            keeper = keepers[key]
            embedding = embeddings[key]
            sniff_data[key] = get, (embedding, time_labels, keeper, sniffer, key)
        return sniff_data

    def get_embeddings_data(self, embeddings, time_labels, classifiers):
        classification_data = {}

        for key, classifier in classifiers.items():
            classification_data[key] = \
                lambda _embedding, _time_labels, _classifier: \
                    (_classifier(_embedding)[0], _time_labels), \
                (embeddings[key], time_labels, classifier)
        return classification_data

    def get_mel_spectrograms(self, *wavs):
        return (self.get_mel_spectrogram(wav) for wav in wavs)

    def get_reconstruction_data(self, wav_generated, wav):
        return {
            'reconstruction': {
                'wave': (wav_generated, wav),
                'mel': (self.get_mel_spectrograms, (wav_generated, wav))
            }
        }

    def get_discriminator_data(self, wav_generated, wav, discriminator):
        def get(_wav_generated, _wav, _discriminator):
            wav_all_r = _discriminator(_wav)[0][0]
            wav_d_r, wav_sub_d_r = wav_all_r
            wav_all_g_detach = _discriminator(_wav_generated)[0][0]
            wav_d_g_detach, wav_sub_d_g_detach = wav_all_g_detach

            wav_d_diff = wav_d_r - wav_d_g_detach
            wav_sub_d_diff = [sub_d_r - sub_d_g for sub_d_r, sub_d_g in zip(wav_sub_d_r, wav_sub_d_g_detach)]

            return [wav_d_diff, wav_sub_d_diff],

        return get, (wav_generated, wav, discriminator)

    def get_adversarial_data(self, wav_generated, wav, discriminator):
        def get(_wav_generated, _wav, _discriminator):
            wav_mom_r, wav_fmap_r = _discriminator(_wav)
            wav_all_r = wav_mom_r[0]
            wav_d_r, wav_sub_d_r = wav_all_r
            wav_mom_g, wav_fmap_g = _discriminator(_wav_generated)
            wav_all_g = wav_mom_g[0]
            wav_d_g, wav_sub_d_g = wav_all_g

            wav_d_diff = wav_d_r - wav_d_g
            wav_sub_d_diff = [sub_d_r - sub_d_g for sub_d_r, sub_d_g in zip(wav_sub_d_r, wav_sub_d_g)]
            return {
                'disc': ([wav_d_diff, wav_sub_d_diff],),
                'fmap': (wav_fmap_g, wav_fmap_r)
            }

        data = {
            'adversarial': {
                'discriminator': (get, (wav_generated, wav, discriminator))
            }
        }
        return data


def main():
    print('Initializing Training Process...')

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='')

    a = parser.parse_args()

    with open(a.config) as f:
        data = f.read()

    config_dict = json.loads(data)

    experiment_name = config_dict['experiment_name']
    tb_logger = pl_loggers.TensorBoardLogger(
        '/mount/logs/',
        name=experiment_name,
        version=config_dict['version'],
        default_hp_metric=False)
    log_dir = Path(tb_logger.log_dir)
    if not log_dir.exists():
        log_dir.mkdir()

    previous_config = os.path.join(tb_logger.log_dir, 'config.yaml')
    h_dict = do_and_cache(lambda: config_dict, previous_config)
    h = AttrDict(h_dict)

    train_dataset = MultilabelWaveDataset(
        base_dir='/datasets',
        dir='/datasets/training_audio',
        name='train',
        config_path='**/train_data_config/*.json',
        segment_size=h.segment_size,
        sampling_rate=h.sampling_rate,
        embedding_size=h.embedding_size,
        augmentation_config=config_dict['augmentation']
    )

    validation_dataset = MultilabelWaveDataset(
        base_dir='/datasets',
        dir='/datasets/training_audio',
        name='train',
        config_path='**/train_data_config/*.json',
        segment_size=h.validation_segment_size,
        sampling_rate=h.sampling_rate,
        embedding_size=h.embedding_size,
        augmentation_config=config_dict['augmentation'],
        deterministic=True,
        size=h.validation_amount
    )

    test_dataset = MultilabelWaveDataset(
        base_dir='/datasets',
        dir='/datasets/training_audio',
        name='test',
        config_path='**/test_data_config/*.json',
        segment_size=h.validation_segment_size,
        sampling_rate=h.sampling_rate,
        embedding_size=h.embedding_size,
        deterministic=True,
        augmentation_config=config_dict['augmentation']
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

    model_configs_dir = os.path.join(tb_logger.log_dir, 'model_configs')
    model_configs_dir_path = Path(model_configs_dir)
    if not model_configs_dir_path.exists():
        model_configs_dir_path.mkdir()

    generator = get_module_from_config(
        do_and_cache(
            lambda: get_static_generator_config(
                initial_skip_ratio=h.initial_skip_ratio,
                expansion_size=h.gen_expansion
            ),
            os.path.join(model_configs_dir, 'generator.yaml')
        )
    )
    for p in generator.parameters():
        p.data.fill_(1e-8)
    print(f'generator:')
    summary(generator,
            input_size=(1, h.segment_size),
            batch_size=h.batch_size,
            device='cpu')

    discriminator_config = do_and_cache(
        lambda: get_static_all_in_one_discriminator(
            expansion_size=h.disc_expansion
        ),
        os.path.join(model_configs_dir, 'discriminator.yaml')
    )

    discriminator = get_module_from_config(discriminator_config)
    discriminator_copy = get_module_from_config(discriminator_config)
    print(f'discriminator:')
    summary(discriminator,
            input_size=(1, h.segment_size),
            batch_size=h.batch_size,
            device='cpu')

    smart_classifier_hiddens = [1092, 546, 364]
    dumb_classifier_hiddens = []

    example_item = train_dataset.label_option_groups
    embedding_dims = (h.embedding_size * h.gen_expansion) // 2
    keepers = generate_keepers_by_example(
        embedding_dims, example_item,
        lambda k, x: do_and_cache(x, os.path.join(model_configs_dir, f'{k}_keeper.yaml')),
        hiddens=smart_classifier_hiddens
    )
    hunters = generate_hunters_by_example(
        embedding_dims, example_item,
        lambda k, x: do_and_cache(x, os.path.join(model_configs_dir, f'{k}_hunter.yaml')),
        hiddens=smart_classifier_hiddens
    )
    hunters_copies = generate_hunters_by_example(
        embedding_dims, example_item,
        lambda k, x: do_and_cache(x, os.path.join(model_configs_dir, f'{k}_hunter.yaml'))
    )

    for key, keeper in keepers.items():
        print(f'{key} keeper:')
        summary(keeper,
                input_size=(embedding_dims, h.segment_size // h.embedding_size),
                batch_size=h.batch_size,
                device='cpu')

    for key, hunter in hunters.items():
        print(f'{key} hunter:')
        summary(hunter,
                input_size=(embedding_dims, h.segment_size // h.embedding_size),
                batch_size=h.batch_size,
                device='cpu')

    sniffers_version = 1
    sniffers_experiment = 'default'
    sniffers_checkpoint = 'best'
    sniffers_log_dirs = {
        key: f'/mount/sniffers/logs/{key}/{sniffers_experiment}/version_{sniffers_version}/' for key in example_item
    }
    sniffers = generate_sniffers_by_example(
        example_item,
        lambda k, x: do_and_cache(x, os.path.join(sniffers_log_dirs[k], 'model_configs', f'{k}_sniffer.yaml')),
        hiddens=smart_classifier_hiddens
    )
    # sniffers = torch.nn.ModuleDict({
    #     key: LabelBiasSniffer.load_from_checkpoint(
    #         os.path.join(sniffers_log_dirs[key], f'checkpoints/{sniffers_checkpoint}'),
    #         sniffers=sniffers, sniffer_key=key
    #     ).sniffer
    #     for key in sniffers.keys()
    # })

    for key, sniffer in sniffers.items():
        print('{} sniffer:'.format(key))
        input_size = InputSize(
            {label: (h.segment_size // h.embedding_size,) for label, value in example_item[key].items()})
        summary(sniffer,
                input_size=input_size,
                batch_size=h.batch_size,
                dtypes={label: RandInt(type=torch.LongTensor, high=value)
                        for label, value in example_item[key].items()},
                device='cpu')

    # model
    model = GanAutoencoder(
        generator,
        discriminator, discriminator_copy,
        keepers,
        hunters, hunters_copies,
        sniffers,
        h, a, h_dict
    )

    # training
    trainer = create_trainer(
        model=model,
        logger=tb_logger,
        intervals={
            'train': h.accumulated_grad_batches,
            'validation': h.accumulated_grad_batches * 10
        },
        h=h
    )
    trainer.fit(model, train_loader, validation_loader)


def create_trainer(model, logger, intervals, h):
    best_checkpoint_callback = BestCheckpointCallback()
    return pl.Trainer(
        gpus=1,
        num_nodes=1,
        precision=32,
        max_steps=1000000,
        logger=logger,
        val_check_interval=intervals['validation'],
        num_sanity_val_steps=h.visualizations_amount,
        callbacks=[
            GlobalSyncCallback(),
            HistoryCheckpointCallback(),
            ContinuousCheckpointCallback(intervals['validation']),
            best_checkpoint_callback,
            ManualOptimizationCallback(h.accumulated_grad_batches, h.gradient_clip, scheduler_args=(model,)),
            OutputSumCallback(
                intervals,
                reset_callbacks=[
                    OutputLoggingCallback(),
                    best_checkpoint_callback
                ]
            ),
            ValveDecayCallback(
                valves_config=h.valves,
                valves_steps=h.accumulated_grad_batches,
                initial_value=h.initial_skip_ratio
            ),
            GanValidationVisualizationCallback(h.visualizations_amount),
            # GanModelsGraphVisualizationCallback()
        ]
    )


if __name__ == '__main__':
    main()
