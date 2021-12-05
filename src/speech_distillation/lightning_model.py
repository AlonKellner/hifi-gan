import os
import warnings
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

from src.speech_distillation.best_checkpoint_callback import BestCheckpointCallback
from src.speech_distillation.confusion_logging_callback import ConfusionLoggingCallback
from src.speech_distillation.gan_models_graph_visualization_callback import GanModelsGraphVisualizationCallback
from src.speech_distillation.global_sync_callback import GlobalSyncCallback
from src.speech_distillation.global_sync_lr_scheduler import GlobalSyncExponentialLR
from src.speech_distillation.label_bias_sniffer import generate_sniffers_by_example
from src.speech_distillation.output_sum_callback import OutputSumCallback
from src.speech_distillation.validation_classification_callback import ValidationClassificationCallback
from src.speech_distillation.yaml_utils import do_and_cache, do_and_cache_dict

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
from src.speech_distillation.validation_visualization_callback import \
    ValidationVisualizationCallback

from torchsummary import summary, InputSize

from static_configs import get_static_generator_configs, \
    get_static_all_in_one_discriminator
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
                 config, args, config_dict):
        super().__init__()
        self.automatic_optimization = False
        self.accumulated_grad_batches = config.accumulated_grad_batches

        self.learning_rates = config_dict['learning_rates']
        self.loss_factors = config_dict['loss_factors']
        self.loss_backward = config_dict['loss_backward']
        self.losses = get_losses_by_types(config_dict['loss_funcs'])

        self.generator = generator
        self.encoder = encoder
        self.decoder = decoder
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
        self.label_weights = label_weights

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

    def transform_data(self, data):
        output = {}

        wav_generated, wav_truth = data['generator']['reconstruction']['wav']
        wav_diff = wav_generated - wav_truth
        output['wav'] = {
            'truth': wav_truth,
            'generated': wav_generated,
            'diff': wav_diff
        }

        mel_generated, mel_truth = data['generator']['reconstruction']['mel']
        mel_diff = mel_generated - mel_truth
        mel_wav_diff = self.get_mel_spectrogram(wav_diff)
        output['mel'] = {
            'truth': mel_truth,
            'generated': mel_generated,
            'diff': mel_diff,
            'wav_diff': mel_wav_diff
        }

        disc = data['discriminator']
        main_generated, main_truth = disc['main']['generated'], disc['main']['truth']
        main_diff = main_generated - main_truth
        sub_generated, sub_truth = disc['sub']['generated'], disc['sub']['truth']
        sub_diff = [sub_g - sub_t for sub_g, sub_t in zip(sub_generated, sub_truth)]
        output['disc'] = {
            'main': {
                'generated': main_generated,
                'on-truth': main_truth,
                'diff': main_diff
            },
            'sub': {
                'generated': sub_generated,
                'on-truth': sub_truth,
                'diff': sub_diff
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

        # output['embedding'] = {} TODO
        return output

    def training_step(self, train_batch, batch_idx):
        losses_dict, _ = self.get_losses(train_batch, batch_idx, backprop=True, get_data=False)
        self._update_discriminator_copy()
        self._update_hunters_copies()
        return losses_dict

    def validation_step(self, val_batch, batch_idx):
        losses_dict, data_dict = self.get_losses(val_batch, batch_idx, backprop=False, get_data=True)
        return losses_dict, self.transform_data(data_dict)

    def get_losses(self, batch, batch_idx, backprop=True, get_data=True):
        wav, wav_path, time_labels, labels = batch
        wav = wav.unsqueeze(1)

        embeddings = self.encoder(wav)
        wav_generated = self.decoder(embeddings)

        detached_embeddings = self._detach_recursively(embeddings)
        detached_wav_generated = self._detach_recursively(wav_generated)

        reconstruction_data = self.get_reconstruction_data(wav_generated, wav)
        adversarial_discriminator_data = self.get_adversarial_data(
            wav_generated, wav,
            self.discriminator_copy
        )
        keepers_data = {'keepers': self.get_embeddings_data(
            embeddings, time_labels, self.label_weights,
            self.keepers
        )}
        adversarial_hunters_data = self.get_adversarial_hunting_data(
            embeddings, detached_embeddings, time_labels, self.label_weights,
            self.hunters_copies, self.keepers, self.sniffers
        )
        generator_data = self._merge_dicts(
            reconstruction_data, adversarial_discriminator_data, keepers_data, adversarial_hunters_data
        )

        discriminator_data = self.get_discriminator_data(
            detached_wav_generated, wav,
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
        return all_losses, out_data

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
            return {key: self._detach_recursively(loss) for key, loss in losses.items()}
        elif isinstance(losses, torch.Tensor):
            if cpu:
                losses = losses.cpu()
            losses = losses.detach()
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
            self.config.n_fft,
            self.config.num_mels,
            self.config.sampling_rate,
            self.config.hop_size,
            self.config.win_size,
            self.config.fmin,
            self.config.fmax_for_loss)

    def get_adversarial_hunting_data(self, embeddings, detached_embeddings, time_labels, weights, hunters, keepers, sniffers):
        def get(_embedding, _detached_embedding, _time_labels, _weights, _hunter, _keeper, _sniffer, _key):
            with torch.no_grad():
                time_labels_keeper_predictions = _keeper(_detached_embedding)['mean']
                bias_labels = _sniffer(time_labels_keeper_predictions[_key])['mean']

                time_labels_keeper_predictions = self._detach_recursively(time_labels_keeper_predictions)
                bias_labels = self._detach_recursively(bias_labels)
            time_labels_predictions = _hunter(_embedding)['mean']
            return {
                label: {
                    label2: (time_labels_predictions[label][label2], bias2, _time_labels[label][label2], _weights[label][label2])
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

    def get_reconstruction_data(self, wav_generated, wav):
        return {
            'reconstruction': {
                'wav': (wav_generated, wav),
                'mel': (self.get_mel_spectrograms, (wav_generated, wav))
            }
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
            wav_r = _discriminator(_wav)
            wav_mom_r, wav_fmap_r = wav_r['output'], wav_r['features']
            del wav_r
            wav_all_r = wav_mom_r['mean']
            wav_d_r, wav_sub_d_r = wav_all_r['output'], wav_all_r['features']
            del wav_all_r
            wav_g = _discriminator(_wav_generated)
            wav_mom_g, wav_fmap_g = wav_g['output'], wav_g['features']
            del wav_g
            wav_all_g = wav_mom_g['mean']
            wav_d_g, wav_sub_d_g = wav_all_g['output'], wav_all_g['features']
            del wav_all_g

            return {
                'disc': {
                    'main': {
                        'truth': wav_d_r,
                        'generated': -wav_d_g
                    },
                    'sub': {
                        'truth': wav_sub_d_r,
                        'generated': [-lo for lo in wav_sub_d_g]
                    }
                },
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
    log_dir.mkdir(parents=True, exist_ok=True)

    previous_config = os.path.join(tb_logger.log_dir, 'config.yaml')
    h_dict = do_and_cache(lambda: config_dict, previous_config)
    h = AttrDict(h_dict)

    set_debug_apis(h.debug)

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

    generator_modules = get_modules_from_configs(
        do_and_cache_dict(
            lambda: get_static_generator_configs(
                expansion_size=h.gen_expansion
            ),
            os.path.join(model_configs_dir, '{}.yaml')
        )
    )
    encoder, decoder = generator_modules['encoder'], generator_modules['decoder']
    generator = torch.nn.Sequential(encoder, decoder)
    for p in generator.parameters():
        p.data.copy_(p.data * h.gen_init_scale)
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

    smart_classifier_hiddens = [546, 364]
    smart_classifier_groups = [13, 1, 1, 1]
    dumb_classifier_hiddens = [546, 364]
    dumb_classifier_groups = [13, 1, 1, 1]

    example_item = train_dataset.label_options_groups
    embedding_dims = (h.embedding_size * h.gen_expansion) // 2
    keepers = generate_keepers_by_example(
        embedding_dims, example_item,
        lambda k, x: do_and_cache(x, os.path.join(model_configs_dir, f'{k}_keeper.yaml')),
        hiddens=dumb_classifier_hiddens,
        groups=dumb_classifier_groups
    )

    for key, keeper in keepers.items():
        print(f'{key} keeper:')
        summary(keeper,
                input_size=(embedding_dims, h.segment_size // h.embedding_size),
                batch_size=h.batch_size,
                device='cpu')

    hunters = generate_hunters_by_example(
        embedding_dims, example_item,
        lambda k, x: do_and_cache(x, os.path.join(model_configs_dir, f'{k}_hunter.yaml')),
        hiddens=smart_classifier_hiddens,
        groups=smart_classifier_groups
    )
    hunters_copies = generate_hunters_by_example(
        embedding_dims, example_item,
        lambda k, x: do_and_cache(x, os.path.join(model_configs_dir, f'{k}_hunter.yaml')),
        hiddens=smart_classifier_hiddens,
        groups=smart_classifier_groups
    )

    for key, hunter in hunters.items():
        print(f'{key} hunter:')
        summary(hunter,
                input_size=(embedding_dims, h.segment_size // h.embedding_size),
                batch_size=h.batch_size,
                device='cpu')

    sniffers = generate_sniffers_by_example(
        example_item,
        lambda k, x: do_and_cache(x, os.path.join(model_configs_dir, f'{k}_sniffer.yaml')),
        hiddens=smart_classifier_hiddens,
        groups=[1, 1, 1, 1, 1]
    )

    for key, sniffer in sniffers.items():
        print('{} sniffer:'.format(key))
        input_size = InputSize(
            {label: (len(value), h.segment_size // h.embedding_size) for label, value in example_item[key].items()}
        )
        summary(sniffer,
                input_size=input_size,
                batch_size=h.batch_size,
                device='cpu')

    # model
    model = GanAutoencoder(
        generator, encoder, decoder,
        discriminator, discriminator_copy,
        keepers,
        hunters, hunters_copies,
        sniffers,
        train_dataset.label_weights_groups,
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
        h=h, h_dict=h_dict
    )
    trainer.fit(model, train_loader, validation_loader)


def create_trainer(model, logger, intervals, h, h_dict):
    best_checkpoint_callback = BestCheckpointCallback()
    callbacks = [
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
        ValidationVisualizationCallback({'few': h.visualizations_amount, 'once': 1}),
        GanModelsGraphVisualizationCallback(),
        ValidationClassificationCallback(intervals['validation'], reset_callbacks=[
            ConfusionLoggingCallback()
        ]),
    ]
    return pl.Trainer(
        gpus=1,
        num_nodes=1,
        precision=32,
        max_steps=100000,
        logger=logger,
        val_check_interval=intervals['validation'],
        num_sanity_val_steps=h.visualizations_amount,
        callbacks=callbacks
    )


def set_debug_apis(state: bool = False):
    torch.autograd.profiler.profile(enabled=state)
    torch.autograd.profiler.emit_nvtx(enabled=state)
    torch.autograd.set_detect_anomaly(mode=state)


if __name__ == '__main__':
    main()
