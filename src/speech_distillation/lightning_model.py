import os
import random
import shutil
import warnings
from pathlib import Path
import numpy as np

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

from src.speech_distillation.best_checkpoint_callback import BestCheckpointCallback
from src.speech_distillation.config_utils import parse_layers
from src.speech_distillation.confusion_logging_callback import ConfusionLoggingCallback
from src.speech_distillation.cycle_calculator import calculate_cycles
from src.speech_distillation.gan_models_graph_visualization_callback import GanModelsGraphVisualizationCallback
from src.speech_distillation.global_sync_callback import GlobalSyncCallback
from src.speech_distillation.global_sync_lr_scheduler import GlobalSyncExponentialLR
from src.speech_distillation.label_bias_sniffer import generate_sniffers_by_example
from src.speech_distillation.output_sum_callback import OutputSumCallback
from src.speech_distillation.recursive_utils import get_recursive
from src.speech_distillation.tensor_utils import mix, expand, unmix
from src.speech_distillation.validation_classification_callback import ValidationClassificationCallback
from src.speech_distillation.yaml_utils import do_and_cache, do_and_cache_dict

warnings.simplefilter(action='ignore', category=FutureWarning)
import argparse
import json
import torch
from torch.utils.data import DataLoader
from multilabel_wave_dataset import MultilabelWaveDataset
from src.meldataset import mel_spectrogram
from src.speech_distillation.continuous_checkpoint_callback import ContinuousCheckpointCallback
from src.speech_distillation.history_checkpoint_callback import HistoryCheckpointCallback
from src.speech_distillation.output_logging_callback import OutputLoggingCallback
from src.speech_distillation.manual_optimization_callback import ManualOptimizationCallback
from src.speech_distillation.validation_visualization_callback import \
    ValidationVisualizationCallback

from torchsummary import summary, InputSize

from static_configs import get_generator_configs, \
    get_discriminator_config
from configurable_module import get_module_from_config, get_modules_from_configs
from custom_losses import recursive_loss, get_losses_by_types

torch.backends.cudnn.benchmark = True

from embedding_classifiers.embedding_classifiers_static_configs import generate_keepers_by_example, \
    generate_hunters_by_example


class GanAutoencoder(pl.LightningModule):
    def __init__(self,
                 generator, encoder, decoder,
                 discriminator, discriminator_copy,
                 keepers,
                 hunters, hunters_copies,
                 sniffers,
                 label_weights,
                 embedding_size, config):
        super().__init__()
        self.automatic_optimization = False

        self.random = random.Random()

        self.optimizers_config = config['learning']['optimizers']
        self.loss_factors = config['learning']['loss_factors']
        self.loss_backward = config['learning']['loss_backward']
        self.losses = get_losses_by_types(config['learning']['loss_funcs'])

        self.generator = generator
        self.encoder = encoder
        self.decoder = decoder
        self.discriminator = discriminator
        self.discriminator_copy = discriminator_copy
        self.keepers = keepers
        self.hunters = hunters
        self.hunters_copies = hunters_copies
        self.sniffers = sniffers
        self.learning_models = torch.nn.ModuleDict({
            'generator': self.generator,
            'discriminator': self.discriminator,
            'keepers': self.keepers,
            'hunters': self.hunters,
            'sniffers': self.sniffers
        })
        self.flat_learning_models, self.flat_optimizers_config = self._create_flat_models_module(self.learning_models, self.optimizers_config)
        self.register_dict(self.flat_learning_models)
        self.label_weights = label_weights

        self.loop_configs = config['loops']
        for loop_type, loop_type_config in self.loop_configs.items():
            batch_size = loop_type_config['batch_size']
            loop_type_config['rolls'] = (batch_size, *calculate_cycles(batch_size, loop_type_config['mix_size']))

        self.learning_config = config['learning']
        self.mel_config = config['mel']
        self.sampling_rate = config['sampling_rate']
        self.embedding_size = embedding_size

        self._update_discriminator_copy()
        self._update_hunters_copies()

    def register_dict(self, flat_models):
        for key, model in flat_models.items():
            self.__setattr__(key, model)

    def _update_discriminator_copy(self):
        self.discriminator_copy.load_state_dict(self.discriminator.state_dict())

    def _update_hunters_copies(self):
        for key, hunter in self.hunters.items():
            self.hunters_copies[key].load_state_dict(hunter.state_dict())

    def _create_flat_models_module(self, models, optimizers_config):
        flat_models, flat_optimizers_config = self._create_flat_models(models, optimizers_config)
        return torch.nn.ModuleDict(flat_models), flat_optimizers_config

    def _create_flat_models(self, models, optimizers_config: {str: object}):
        if isinstance(models, (dict, torch.nn.ModuleDict)):
            key_groups = {key: key.split(',') for key in optimizers_config.keys()}
            grouped_models = {key: models[sub_keys[0]] if len(sub_keys) == 1 else torch.nn.ModuleList([models[sub_key] for sub_key in sub_keys]) for key, sub_keys in key_groups.items()}
            a_2d_array = {key: self._create_flat_models(pair, optimizers_config[key]) for key, pair in
                               grouped_models.items()}
            models_2d_array = {key: pair[0] for key, pair in a_2d_array.items()}
            configs_2d_array = {key: pair[1] for key, pair in a_2d_array.items()}
            flat_models = {
                (f'{key_2d}' if key_1d is None else f'{key_2d}/{key_1d}'): model
                for key_2d, models_1d_array in models_2d_array.items()
                for key_1d, model in models_1d_array.items()
            }
            flat_config = {
                (f'{key_2d}' if key_1d is None else f'{key_2d}/{key_1d}'): optimizer_config
                for key_2d, optimizers_1d_array in configs_2d_array.items()
                for key_1d, optimizer_config in optimizers_1d_array.items()
            }
            return flat_models, flat_config
        else:
            return {None: models}, {None: optimizers_config}

    def get_learning_models(self):
        return self.flat_learning_models

    def forward(self, x):
        wav, wav_path, time_labels, labels = x
        wav = wav.unsqueeze(1)
        embeddings = self.encoder(wav)
        wav_generated = self.decoder(embeddings)
        wav_diff = wav - wav_generated
        wav_generated_mel, wav_mel, wav_diff_mel = self.get_mel_spectrograms(wav_generated, wav, wav_diff)
        wav_mel_diff_inverse = wav_mel - wav_generated_mel

        wav_discrimination = self.discriminator(wav)['output']['mean']['output']
        wav_generated_discrimination = self.discriminator(wav_generated)['output']['mean']['output']

        keepers_outputs = {key: self.keepers[key](value) for key, value in embeddings.items()}
        sniffers_outputs = {key: self.sniffers[key](value['mean'][key]) for key, value in keepers_outputs.items()}
        hunters_outputs = {key: self.hunters[key](value) for key, value in embeddings.items()}

        results = {
            'wavs': {
                'original': wav,
                'generated': wav_generated,
                'diff': wav_diff
            },
            'mels': {
                'original': wav_mel,
                'generated': wav_generated_mel,
                'diff': wav_diff_mel,
                'inverse_diff': wav_mel_diff_inverse,
            },
            'discs': {
                'original': wav_discrimination,
                'generated': wav_generated_discrimination
            },
            'labels': {
                'keeps': keepers_outputs,
                'sniffs': sniffers_outputs,
                'hunts': hunters_outputs
            },
        }
        results = self._detach_recursively(results)
        return results

    def configure_optimizers(self):
        optimizers = self._models_to_optimizers(self.get_learning_models(), self.flat_optimizers_config)

        schedulers = [
            GlobalSyncExponentialLR(optimizer, lr_decay=self.learning_config['lr_decay'], global_step=self.global_step) for
            optimizer in optimizers
        ]
        return optimizers, schedulers

    def _models_to_optimizers(self, models, optimizers_config):
        if isinstance(models, (dict, torch.nn.ModuleDict)):
            sorted_keys = list(models.keys())
            sorted_keys.sort()
            optimizers_2d_array = [self._models_to_optimizers(models[key], optimizers_config[key]) for key in sorted_keys]
            flat_optimizers = [
                optimizer
                for optimizers_1d_array in optimizers_2d_array
                for optimizer in optimizers_1d_array
            ]
            return flat_optimizers
        else:
            return [torch.optim.AdamW(
                models.parameters(),
                lr=optimizers_config,
                betas=(self.learning_config['adam_b1'], self.learning_config['adam_b2']),
                amsgrad=True
            )]

    def _extract_labels(self, all_args, index):
        result = {
            key1: {
                key2: {
                    key3: args3[index]
                    for key3, args3 in args2.items()
                } for key2, args2 in args1.items()
            } for key1, args1 in all_args.items()
        }
        result = {group_name2: group2 for group_name, group in result.items()
                  for group_name2, group2 in group.items()}
        return result

    def transform_data(self, data, extra_data):
        output = {}

        wav_generated, wav_truth = extra_data['wav']['generated'], extra_data['wav']['truth']
        output['wav'] = {
            'truth': wav_truth,
            'generated': wav_generated
        }

        mel_truth, mel_generated = self.get_mel_spectrograms(wav_truth, wav_generated)
        output['mel'] = {
            'truth': mel_truth,
            'generated': mel_generated
        }

        emb_generated, emb_truth = data['generator']['emb_recon']
        output['embedding'] = {
            'on-truth': emb_truth,
            'generated': emb_generated
        }

        disc = data['discriminator']
        main_generated, main_truth = disc['main']['generated'], disc['main']['truth']
        sub_generated, sub_truth = disc['sub']['generated'], disc['sub']['truth']
        output['disc'] = {
            'main': {
                'generated': main_generated,
                'on-truth': main_truth
            },
            'sub': {
                'generated': sub_generated,
                'on-truth': sub_truth
            }
        }

        hunters = data['hunters']
        keepers = data['generator']['keepers']
        sniffers = data['sniffers']
        label_truth = self._extract_labels(hunters, 1)
        label_keep = self._extract_labels(keepers, 0)
        label_sniff = self._extract_labels(sniffers, 0)
        label_hunt = self._extract_labels(hunters, 0)
        output['label'] = {
            'truth': label_truth,
            'keep': label_keep,
            'sniff': label_sniff,
            'hunt': label_hunt
        }
        return output

    def cut_and_roll(self, tensor, size, cut_dim, roll_dim):
        length = tensor.size(cut_dim)
        narrowed_tensors = torch.split(tensor, [size, length-size], dim=cut_dim)
        rolled_tensors = [torch.roll(narrowed_tensor, roll, dims=roll_dim) for roll, narrowed_tensor in
                          enumerate(narrowed_tensors)]
        cat_tensor = torch.cat(rolled_tensors, dim=cut_dim)
        return cat_tensor

    def cut_and_roll_batch(self, batch, roll_dim=0, cut_dim=1):
        wav, wav_path, time_labels, labels = batch

        batch_size, length = wav.size()
        embedded_length = length // self.embedding_size
        embedded_cut = self.random.randint(0, embedded_length)
        wav_cut = embedded_cut * self.embedding_size

        wav = self.cut_and_roll(wav, wav_cut, cut_dim, roll_dim)
        time_labels = get_recursive(self.cut_and_roll, time_labels, args=[embedded_cut, cut_dim, roll_dim])

        return wav, wav_path, time_labels, labels

    def training_step(self, train_batch, batch_idx):
        # train_batch = self.cut_and_roll_batch(train_batch)
        losses_dict, _, _ = self.get_losses(train_batch, batch_idx, 'train', backprop=True, get_data=False)
        self._update_discriminator_copy()
        self._update_hunters_copies()
        return losses_dict

    def validation_step(self, val_batch, batch_idx):
        losses_dict, data_dict, extra_data_dict = self.get_losses(val_batch, batch_idx, 'validation', backprop=False,
                                                                  get_data=True)
        return losses_dict, self.transform_data(data_dict, extra_data_dict)

    def get_losses(self, batch, batch_idx, data_type, backprop=True, get_data=True):
        wav_narrow, wav_path, time_labels, labels = batch
        wav_narrow = wav_narrow.unsqueeze(1)

        embeddings_narrow = self.encoder(wav_narrow)
        mix_key = next(iter(embeddings_narrow.keys()))

        wav_order = self.expand_tensor(wav_narrow, data_type)
        embeddings_order = self.expand_tensor(embeddings_narrow, data_type)

        embeddings_mix = embeddings_order.copy()
        embeddings_mix[mix_key] = self.mix_embedding(embeddings_mix[mix_key], data_type)

        wav_generated_mix = self.decoder(embeddings_mix)
        wav_generated_narrow = torch.narrow(
            wav_generated_mix, dim=0, start=0, length=self.loop_configs[data_type]['batch_size']
        )

        embeddings_generated_mix = self.encoder(wav_generated_mix)

        embeddings_generated_order = embeddings_generated_mix.copy()
        embeddings_generated_order[mix_key] = self.unmix_embedding(embeddings_generated_mix[mix_key], data_type)

        wav_regenerated_order = self.decoder(embeddings_generated_order)

        wav_generated_all = torch.cat([wav_regenerated_order, wav_generated_mix], dim=0)
        wav_generated_order_all = torch.cat([wav_regenerated_order, wav_generated_narrow], dim=0)
        wav_order_all = torch.cat([wav_order, wav_narrow], dim=0)

        detached_embeddings = self._detach_recursively(embeddings_narrow)
        detached_wav_generated_narrow = self._detach_recursively(wav_generated_narrow)
        detached_wav_generated_all = self._detach_recursively(wav_generated_all)

        raw_recon_data = self.get_raw_reconstruction_data(wav_generated_order_all, wav_order_all)
        emb_recon_data = self.get_embedding_reconstruction_data(embeddings_generated_order, embeddings_order)
        adversarial_discriminator_data = self.get_adversarial_data(
            wav_generated_all, wav_order_all,
            self.discriminator_copy
        )
        keepers_data = {'keepers': self.get_embeddings_data(
            embeddings_narrow, time_labels, self.label_weights,
            self.keepers
        )}
        adversarial_hunters_data = self.get_adversarial_hunting_data(
            embeddings_narrow, detached_embeddings, time_labels, self.label_weights,
            self.hunters_copies, self.keepers, self.sniffers
        )
        generator_data = self._merge_dicts(
            raw_recon_data, emb_recon_data, adversarial_discriminator_data, keepers_data, adversarial_hunters_data
        )

        discriminator_data = self.get_discriminator_data(
            detached_wav_generated_all, wav_narrow,
            self.discriminator
        )

        hunters_data = self.get_embeddings_data(
            detached_embeddings, time_labels, self.label_weights,
            self.hunters
        )

        sniffers_data = self.get_sniffers_data(
            detached_embeddings, time_labels, self.label_weights,
            self.keepers, self.sniffers
        )

        all_data = {
            'generator': generator_data,
            'discriminator': discriminator_data,
            'hunters': hunters_data,
            'sniffers': sniffers_data
        }
        if get_data:
            all_losses, losses_sum, out_data = self._calculate_losses_and_data(
                self.losses, self.loss_factors,
                self.loss_backward, all_data,
                backprop=backprop)
        else:
            out_data = None
            all_losses, losses_sum = self._calculate_losses(
                self.losses, self.loss_factors,
                self.loss_backward, all_data,
                backprop=backprop)
        extra_data = {'wav': {
            'truth': wav_narrow,
            'generated': wav_generated_all
        }}
        extra_data = self._detach_recursively(extra_data, cpu=True)
        return all_losses, out_data, extra_data

    def unmix_embedding(self, embedding, data_type):
        return get_recursive(
            unmix,
            tensor=embedding,
            kwargs={'rolls': self.loop_configs[data_type]['rolls'], 'dim': 0}
        )

    def mix_embedding(self, embedding, data_type):
        return get_recursive(
            mix,
            tensor=embedding,
            kwargs={'rolls': self.loop_configs[data_type]['rolls'], 'dim': 0}
        )

    def expand_tensor(self, tensor, data_type):
        return get_recursive(
            expand,
            tensor=tensor,
            kwargs={'size': sum(self.loop_configs[data_type]['rolls']), 'dim': 0}
        )

    def _calculate_losses(self, loss, factor, backward, data, backprop=True):
        if isinstance(data, tuple) \
                and len(data) == 2 \
                and callable(data[0]) \
                and isinstance(data[1], tuple):
            func, params = data
            data = func(*params)
        if isinstance(data, dict):
            losses_sum = 0
            losses = {}
            for key in data.keys():
                current_losses, current_sum = self._calculate_losses(
                    loss[key] if isinstance(loss, dict) else loss,
                    factor[key] if isinstance(factor, dict) else factor,
                    backward[key] if isinstance(backward, dict) else False,
                    data[key], backprop=backprop)
                losses[key] = current_losses
                losses_sum = current_sum + losses_sum
            losses['total'] = losses_sum
        else:
            current_loss = recursive_loss(loss, *data)
            current_loss = current_loss * factor
            losses, losses_sum = current_loss, current_loss

        losses = self._detach_recursively(losses, cpu=True)
        if isinstance(backward, bool) and backward:
            if backprop:
                self.manual_backward(losses_sum, retain_graph=True)
            losses_sum = losses_sum.detach().cpu()
        return losses, losses_sum

    def _calculate_losses_and_data(self, loss, factor, backward, data, backprop=True):
        if isinstance(data, tuple) \
                and len(data) == 2 \
                and callable(data[0]) \
                and isinstance(data[1], tuple):
            func, params = data
            data = func(*params)
        if isinstance(data, dict):
            losses_sum = 0
            losses = {}
            for key in data.keys():
                current_losses, current_sum, currunt_data = self._calculate_losses_and_data(
                    loss[key] if isinstance(loss, dict) else loss,
                    factor[key] if isinstance(factor, dict) else factor,
                    backward[key] if isinstance(backward, dict) else False,
                    data[key], backprop=backprop)
                data[key] = currunt_data
                losses[key] = current_losses
                losses_sum = current_sum + losses_sum
            losses['total'] = losses_sum
        else:
            current_loss = recursive_loss(loss, *data)
            current_loss = current_loss * factor
            losses, losses_sum = current_loss, current_loss

        out_data = self._detach_recursively(data, cpu=True)
        losses = self._detach_recursively(losses, cpu=True)
        if isinstance(backward, bool) and backward:
            if backprop:
                self.manual_backward(losses_sum, retain_graph=True)
            losses_sum = losses_sum.cpu().detach()
        return losses, losses_sum, out_data

    def _detach_recursively(self, losses, cpu=False):
        if isinstance(losses, dict):
            return {key: self._detach_recursively(loss, cpu=cpu) for key, loss in losses.items()}
        if isinstance(losses, (list, tuple)):
            return [self._detach_recursively(loss, cpu=cpu) for loss in losses]
        elif isinstance(losses, torch.Tensor):
            losses = losses.detach()
            if cpu:
                losses = losses.cpu()
            return losses
        else:
            return losses

    def _merge_dicts(self, *all_dicts):
        result_dict = {}
        for current_dict in all_dicts:
            result_dict = self._merge_into(result_dict, current_dict)
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
            self.mel_config['n_fft'],
            self.mel_config['num_mels'],
            self.sampling_rate,
            self.mel_config['hop_size'],
            self.mel_config['win_size'],
            self.mel_config['fmin'],
            self.mel_config['fmax'])

    def get_adversarial_hunting_data(self, embeddings, detached_embeddings, time_labels, weights, hunters, keepers,
                                     sniffers):
        def get(_embedding, _detached_embedding, _time_labels, _weights, _hunter, _keeper, _sniffer, _key):
            with torch.no_grad():
                time_labels_keeper_predictions = _keeper(_detached_embedding)['mean']
                bias_labels = _sniffer(time_labels_keeper_predictions[_key])['mean']

                time_labels_keeper_predictions = self._detach_recursively(time_labels_keeper_predictions)
                bias_labels = self._detach_recursively(bias_labels)
            time_labels_predictions = _hunter(_embedding)['mean']
            return {
                label: {
                    label2: (
                        time_labels_predictions[label][label2], bias2, _time_labels[label][label2],
                        _weights[label][label2])
                    for label2, bias2 in bias.items()
                } for label, bias in bias_labels.items()
            }

        hunt_data = {}
        for key, hunter in hunters.items():
            keeper = keepers[key]
            sniffer = sniffers[key]
            embedding = embeddings[key]
            detached_embedding = detached_embeddings[key]
            hunt_data[key] = get, (embedding, detached_embedding, time_labels, weights, hunter, keeper, sniffer, key)

        data = {
            'adversarial': {
                'hunters': hunt_data
            }
        }
        return data

    def get_sniffers_data(self, embeddings, time_labels, weights, keepers, sniffers):
        def get(_embedding, _time_labels, _weights, _keeper, _sniffer, _key):
            with torch.no_grad():
                time_labels_keeper_predictions = _keeper(_embedding)['mean']

                time_labels_keeper_predictions = self._detach_recursively(time_labels_keeper_predictions)
            bias_labels = _sniffer(time_labels_keeper_predictions[_key])['mean']
            return {
                label: {
                    label2: (bias2, _time_labels[label][label2], _weights[label][label2])
                    for label2, bias2 in bias.items()
                } for label, bias in bias_labels.items()
            }

        sniff_data = {}
        for key, sniffer in sniffers.items():
            keeper = keepers[key]
            embedding = embeddings[key]
            sniff_data[key] = get, (embedding, time_labels, weights, keeper, sniffer, key)
        return sniff_data

    def get_embeddings_data(self, embeddings, time_labels, weights, classifiers):
        def get(_embedding, _time_labels, _weights, _classifier):
            classifications = _classifier(_embedding)['mean']
            return {
                key1: {
                    key2: (classification2, _time_labels[key1][key2], _weights[key1][key2])
                    for key2, classification2 in classification.items()
                } for key1, classification in classifications.items()
            }

        classification_data = {}

        for key, classifier in classifiers.items():
            classification_data[key] = get, (embeddings[key], time_labels, weights, classifier)
        return classification_data

    def get_mel_spectrograms(self, *wavs):
        return tuple(self.get_mel_spectrogram(wav) for wav in wavs)

    def get_raw_reconstruction_data(self, wav_generated, wav):
        return {
            'raw_recon': {
                'wav': (wav_generated, wav),
                'mel': (self.get_mel_spectrograms, (wav_generated, wav))
            }
        }

    def get_embedding_reconstruction_data(self, embedding_generated, embedding_wav):
        return {
            'emb_recon': (embedding_generated, embedding_wav)
        }

    def get_discriminator_data(self, wav_generated, wav, discriminator):
        def get(_wav_generated, _wav, _discriminator):
            wav_all_r = _discriminator(_wav)['output']['mean']
            wav_d_r, wav_sub_d_r = wav_all_r['output'], wav_all_r['features']
            del wav_all_r
            wav_all_g_detach = _discriminator(_wav_generated)['output']['mean']
            wav_d_g_detach, wav_sub_d_g_detach = wav_all_g_detach['output'], wav_all_g_detach['features']
            del wav_all_g_detach

            return {
                'main': {
                    'truth': wav_d_r,
                    'generated': -wav_d_g_detach
                },
                'sub': {
                    'truth': wav_sub_d_r,
                    'generated': [-lo for lo in wav_sub_d_g_detach]
                }
            }

        return get, (wav_generated, wav, discriminator)

    def get_adversarial_data(self, wav_generated, wav, discriminator):
        def get(_wav_generated, _wav, _discriminator):
            wav_truth = _discriminator(_wav)
            wav_mom_truth, wav_fmap_truth = wav_truth['output'], wav_truth['features']
            del wav_truth
            wav_all_truth = wav_mom_truth['mean']
            wav_d_truth, wav_sub_d_truth = wav_all_truth['output'], wav_all_truth['features']
            del wav_all_truth
            wav_generated = _discriminator(_wav_generated)
            wav_mom_generated, wav_fmap_generated = wav_generated['output'], wav_generated['features']

            del wav_generated
            wav_all_generated = wav_mom_generated['mean']
            wav_d_generated, wav_sub_d_generated = wav_all_generated['output'], wav_all_generated['features']
            del wav_all_generated

            wav_fmap_generated_clipped = get_recursive(
                torch.narrow,
                wav_fmap_generated,
                kwargs={'dim': 0, 'start': 0, 'length': wav_d_truth.size(0)}
            )

            return {
                'disc': {
                    'main': {
                        'truth': wav_d_truth,
                        'generated': -wav_d_generated
                    },
                    'sub': {
                        'truth': wav_sub_d_truth,
                        'generated': [-lo for lo in wav_sub_d_generated]
                    }
                },
                'fmap': (wav_fmap_generated_clipped, wav_fmap_truth)
            }

        data = {
            'adversarial': {
                'discriminator': (get, (wav_generated, wav, discriminator))
            }
        }
        return data


def create_dataset(name, loop_config, dataset_config, augmentation_config, sampling_rate, embedding_size):
    base_dir = '/datasets'
    dataset = MultilabelWaveDataset(
        data_dir=f'{base_dir}/data',
        aug_dir=f'{base_dir}/aug',
        cache_dir=f'{base_dir}/cache',
        name=name,
        segment_length=loop_config['segment_length'],
        sampling_rate=sampling_rate,
        embedding_size=embedding_size,
        augmentation_config=augmentation_config,
        **dataset_config['dataset']
    )
    loader = DataLoader(
        dataset,
        sampler=None,
        batch_size=loop_config['batch_size'],
        pin_memory=True,
        drop_last=True,
        **dataset_config['loader'])
    return {
        'dataset': dataset,
        'loader': loader
    }


def create_datasets(loops_config, datasets_config, augmentation_config, sampling_rate, embedding_size):
    return {
        key: create_dataset(
            key, loops_config[key], datasets_config[key], augmentation_config, sampling_rate, embedding_size
        )
        for key in loops_config
    }


def main():
    print('Initializing Training Process...')

    config, datasets, model, tb_logger = initialize_model_objects()

    trainer = create_trainer(
        model=model,
        logger=tb_logger,
        intervals={
            'train': config['learning']['accumulated_grad_batches'],
            'validation': config['learning']['accumulated_grad_batches'] * 20
        },
        config=config
    )
    trainer.fit(model, datasets['train']['loader'], datasets['validation']['loader'])


def initialize_model_objects():
    tb_logger, config = create_config()
    parsed_layers = parse_layers(config['models']['generator']['layers'])
    embedding_size = int(np.prod([layer_params[2] for layer_types, layer_params in parsed_layers]))
    set_debug_apis(config['debug'])
    datasets = create_datasets(
        config['loops'], config['data'], config['augmentation'], config['sampling_rate'], embedding_size
    )
    generator, encoder, decoder, \
    discriminator, discriminator_copy, \
    keepers, \
    hunters, hunters_copies, \
    sniffers = create_models(config['models'], config['loops'], datasets, embedding_size, tb_logger)
    model = GanAutoencoder(
        generator, encoder, decoder,
        discriminator, discriminator_copy,
        keepers,
        hunters, hunters_copies,
        sniffers,
        datasets['train']['dataset'].label_weights_groups,
        embedding_size, config
    )
    return config, datasets, model, tb_logger


def create_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='')

    a = parser.parse_args()

    with open(a.config) as f:
        data = f.read()

    config_dict = json.loads(data)

    experiment = config_dict['experiment']
    tb_logger = pl_loggers.TensorBoardLogger(
        '/mount/logs/',
        name=experiment['name'],
        version=experiment['version'],
        default_hp_metric=False)
    log_dir = Path(tb_logger.log_dir)

    source_dir = log_dir
    target_dir = log_dir
    if 'copy' in experiment and experiment['copy']['enabled'] == True:
        if log_dir.exists() and not experiment['overwrite']:
            raise Exception('Cannot copy into existing version when overwrite is false.')
        copy_config = experiment['copy']
        copy_name = copy_config['name'] if 'name' in copy_config else experiment['name']
        copy_version = copy_config['version'] if 'version' in copy_config else experiment['version']
        copy_logger = pl_loggers.TensorBoardLogger(
            '/mount/logs/',
            name=copy_name,
            version=copy_version,
            default_hp_metric=False)
        source_dir = copy_logger.log_dir
        del copy_logger

    if 'overwrite' in experiment and experiment['overwrite'] == True and log_dir.exists():
        shutil.rmtree(log_dir)
        if log_dir.exists():
            log_dir.rmdir()
        if log_dir.exists():
            raise Exception(f'The directory [{log_dir}] still exists! delete it manually.')
    log_dir.mkdir(parents=True, exist_ok=True)
    source_config = os.path.join(source_dir, 'config.yaml')
    target_config = os.path.join(target_dir, 'config.yaml')
    config = do_and_cache(lambda: config_dict, target_config, source_config)
    return tb_logger, config


def create_models(models_config, loops_config, datasets, embedding_size, tb_logger):
    model_configs_dir = os.path.join(tb_logger.log_dir, 'model_configs')
    model_configs_dir_path = Path(model_configs_dir)
    if not model_configs_dir_path.exists():
        model_configs_dir_path.mkdir(parents=True, exist_ok=True)

    decoder, encoder, generator = create_generator(models_config, loops_config, model_configs_dir)

    discriminator, discriminator_copy = create_discriminator(models_config, loops_config, model_configs_dir)

    example_item = datasets['train']['dataset'].label_options_groups
    embedding_dims = (embedding_size * models_config['generator']['expansion']) // 2

    keepers = create_keepers(models_config, loops_config, embedding_dims, embedding_size, example_item, model_configs_dir)

    hunters, hunters_copies = create_hunters(models_config, loops_config, embedding_dims, embedding_size, example_item, model_configs_dir)

    sniffers = create_sniffers(models_config, loops_config, embedding_size, example_item, model_configs_dir)

    return generator, encoder, decoder, \
           discriminator, discriminator_copy, \
           keepers, \
           hunters, hunters_copies, \
           sniffers


def create_sniffers(models_config, loops_config, embedding_size, example_item, model_configs_dir):
    sniffers_layers = parse_layers(models_config['sniffers']['layers'])
    sniffers = generate_sniffers_by_example(
        example_item,
        lambda k, x: do_and_cache(x, os.path.join(model_configs_dir, f'{k}_sniffer.yaml')),
        layers=sniffers_layers
    )
    for key, sniffer in sniffers.items():
        print('{} sniffer:'.format(key))
        input_size = InputSize(
            {label: (len(value), loops_config['train']['segment_length'] // embedding_size) for label, value in
             example_item[key].items()}
        )
        summary(sniffer,
                input_size=input_size,
                batch_size=loops_config['train']['batch_size'],
                device='cpu')
    return sniffers


def create_hunters(models_config, loops_config, embedding_dims, embedding_size, example_item, model_configs_dir):
    hunters_layers = parse_layers(models_config['hunters']['layers'])
    hunters = generate_hunters_by_example(
        embedding_dims, example_item,
        lambda k, x: do_and_cache(x, os.path.join(model_configs_dir, f'{k}_hunter.yaml')),
        layers=hunters_layers
    )
    hunters_copies = generate_hunters_by_example(
        embedding_dims, example_item,
        lambda k, x: do_and_cache(x, os.path.join(model_configs_dir, f'{k}_hunter.yaml')),
        layers=hunters_layers
    )

    for key, hunter in hunters.items():
        print(f'{key} hunter:')
        summary(hunter,
                input_size=(embedding_dims, loops_config['train']['segment_length'] // embedding_size),
                batch_size=loops_config['train']['batch_size'],
                device='cpu')
    return hunters, hunters_copies


def create_keepers(models_config, loops_config, embedding_dims, embedding_size, example_item, model_configs_dir):
    keepers_layers = parse_layers(models_config['keepers']['layers'])
    keepers = generate_keepers_by_example(
        embedding_dims, example_item,
        lambda k, x: do_and_cache(x, os.path.join(model_configs_dir, f'{k}_keeper.yaml')),
        layers=keepers_layers
    )
    for key, keeper in keepers.items():
        print(f'{key} keeper:')
        summary(keeper,
                input_size=(embedding_dims, loops_config['train']['segment_length'] // embedding_size),
                batch_size=loops_config['train']['batch_size'],
                device='cpu')
    return keepers


def create_discriminator(models_config, loops_config, model_configs_dir):
    discriminator_layers = parse_layers(models_config['discriminator']['layers'])
    discriminator_config = do_and_cache(
        lambda: get_discriminator_config(
            expansion_size=models_config['discriminator']['expansion'],
            ensemble_size=models_config['discriminator']['ensemble'],
            layers=discriminator_layers
        ),
        os.path.join(model_configs_dir, 'discriminator.yaml')
    )
    discriminator = get_module_from_config(discriminator_config)
    discriminator_copy = get_module_from_config(discriminator_config)
    print(f'discriminator:')
    summary(discriminator,
            input_size=(1, loops_config['train']['segment_length']),
            batch_size=loops_config['train']['batch_size'],
            device='cpu')
    return discriminator, discriminator_copy


def create_generator(models_config, loops_config, model_configs_dir):
    generator_layers = parse_layers(models_config['generator']['layers'])
    generator_modules = get_modules_from_configs(
        do_and_cache_dict(
            lambda: get_generator_configs(layers=generator_layers,
                                          expansion_size=models_config['generator']['expansion']),
            os.path.join(model_configs_dir, '{}.yaml')
        )
    )
    encoder, decoder = generator_modules['encoder'], generator_modules['decoder']
    generator = torch.nn.Sequential(encoder, decoder)
    print(f'generator:')
    summary(generator,
            input_size=(1, loops_config['train']['segment_length']),
            batch_size=loops_config['train']['batch_size'],
            device='cpu')
    return decoder, encoder, generator


def create_trainer(model, logger, intervals, config):
    best_checkpoint_callback = BestCheckpointCallback()
    callbacks = [
        GlobalSyncCallback(),
        HistoryCheckpointCallback(),
        ContinuousCheckpointCallback(intervals['validation']),
        best_checkpoint_callback,
        ManualOptimizationCallback(config['learning']['accumulated_grad_batches'], config['learning']['gradient_clip'],
                                   scheduler_args=(model,)),
        OutputSumCallback(
            intervals,
            reset_callbacks=[
                OutputLoggingCallback(),
                best_checkpoint_callback
            ]
        ),
        ValidationVisualizationCallback({'few': config['visualize'], 'once': 1}),
        GanModelsGraphVisualizationCallback(),
        ValidationClassificationCallback(intervals['validation'], reset_callbacks=[
            ConfusionLoggingCallback()
        ]),
    ]
    return pl.Trainer(
        gpus=1,
        num_nodes=1,
        precision=32,
        max_steps=1000000,
        logger=logger,
        val_check_interval=intervals['validation'],
        num_sanity_val_steps=config['visualize'],
        callbacks=callbacks
    )


def set_debug_apis(state: bool = False):
    torch.autograd.profiler.profile(enabled=state)
    torch.autograd.profiler.emit_nvtx(enabled=state)
    torch.autograd.set_detect_anomaly(mode=state)


if __name__ == '__main__':
    main()
