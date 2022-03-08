import os.path

import torch
import numpy as np
from pytorch_lightning.callbacks import Callback

from src.utils import plot_spectrogram, plot_categorical, plot_image
from logging_utils import rank

EPSILON = 1e-04

class ValidationVisualizationCallback(Callback):
    def __init__(self, amounts_to_log):
        self.amounts_to_log = amounts_to_log
        self.truth_to_log = {}
        self.to_log = {}

    def on_validation_batch_end(
            self,
            trainer,
            pl_module,
            outputs,
            batch,
            batch_idx: int,
            dataloader_idx: int,
    ):
        losses, outputs = outputs

        sw = pl_module.logger.experiment
        self.visualize(trainer, pl_module, batch, batch_idx, dataloader_idx, sw,
                       'wavs', self._visualize_wav, 'few', outputs['wav'])
        self.visualize(trainer, pl_module, batch, batch_idx, dataloader_idx, sw,
                       'mels', self._visualize_mel, 'few', outputs['mel'])
        self.visualize(trainer, pl_module, batch, batch_idx, dataloader_idx, sw,
                       'labels', self._visualize_label, 'few', outputs['label'], level=1)
        self.visualize(trainer, pl_module, batch, batch_idx, dataloader_idx, sw,
                       'outputs', self._visualize_output, 'once', outputs)
        del outputs
        self.visualize_model_parameters(trainer, pl_module, batch, batch_idx, dataloader_idx,
                                        sw)

    def visualize_model_parameters(self, trainer, pl_module, batch, batch_idx, dataloader_idx, sw):
        if self._check_to_log(batch_idx, 'models', 'once'):
            models = pl_module.get_learning_models()
            for name, model in models.items():
                sw.add_histogram(rank(f'models/{name}'), torch.cat([p.detach().view(-1) for p in model.parameters()]),
                                 pl_module.global_step)
            for name, model in models.items():
                for param_name, param in model.named_parameters():
                    sw.add_histogram(rank(f'models/{name}/{param_name}'), param.detach().view(-1),
                                     pl_module.global_step)
            for name, model in models.items():
                self.visualize_model_parameter_snake_image(trainer, pl_module, batch, batch_idx, dataloader_idx, sw, name, model)

    def visualize_model_parameter_snake_image(self, trainer, pl_module, batch, batch_idx, dataloader_idx, sw, name,
                                              model):
        weights = [(name, param.cpu()) for name, param in model.named_parameters() if 'weight' in name and 'weight_g' not in name]
        biases = [(name, param.cpu()) for name, param in model.named_parameters() if 'bias' in name]

        weights_2d = [(name, torch.mean(weight, dim=tuple(range(2, len(weight.size()))))) for name, weight in weights]
        biases_1d = biases

        snakes = {}
        current_snake = []
        current_snake_names = []
        last_snake_dim = None
        for (weight_name, weight_2d), (bias_name, bias_1d) in zip(weights_2d, biases_1d):
            current_snake_names.append(weight_name)
            current_snake_names.append(bias_name)
            if last_snake_dim is None or last_snake_dim == weight_2d.size(1):
                if weight_2d.size(0) == bias_1d.size(0):
                    current_snake.append((weight_2d, bias_1d))
                    last_snake_dim = bias_1d.size(0)
                else:
                    current_snake.append((weight_2d, None))
                    common_prefix = os.path.commonprefix(current_snake_names)
                    snakes[common_prefix] = current_snake
                    current_snake = []
                    current_snake_names = []
                    last_snake_dim = None
            else:
                common_prefix = os.path.commonprefix(current_snake_names).strip('.')
                snakes[common_prefix] = current_snake
                current_snake = []
                current_snake_names = []
                last_snake_dim = None

        snake_images = {}
        for snake_name, snake in snakes.items():
            values = torch.cat([torch.cat([weight.view(-1), bias.view(-1)]) for weight, bias in snake])
            max_value = torch.max(values).item()
            min_value = torch.min(values).item()
            snake_wide = [(weight, bias) for index, (weight, bias) in enumerate(snake) if index%2==0]
            snake_high = [(weight, bias) for index, (weight, bias) in enumerate(snake) if index%2==1]

            height = snake_wide[0][0].size(1)
            width = 0
            for current_index in range(len(snake_wide)):
                extra_width = 0
                extra_height = 0
                weight_w, bias_w = snake_wide[current_index]
                w_link_width = weight_w.size(0)
                extra_width += w_link_width
                if bias_w is not None:
                    extra_height += 1

                if current_index < len(snake_high):
                    weight_h, bias_h = snake_high[current_index]
                    h_link_width = weight_h.size(1)
                    h_link_height = weight_h.size(0)
                    extra_height += h_link_height
                    if bias_h is not None:
                        extra_width += 1
                height += extra_height
                width += extra_width


            snake_image = np.ones([width, height], dtype=np.float)*min_value
            current_width = 0
            current_height = 0
            for current_index in range(len(snake_wide)):
                weight_w, bias_w = snake_wide[current_index]
                w_link_width = weight_w.size(0)
                w_link_height = weight_w.size(1)
                snake_image[current_width:current_width+w_link_width,current_height:current_height+w_link_height] = weight_w.numpy()
                current_height += w_link_height
                if bias_w is not None:
                    snake_image[:,current_height] = max_value
                    snake_image[current_width:current_width+w_link_width,current_height] = bias_w.numpy()
                    current_height += 1

                if current_index < len(snake_high):
                    weight_h, bias_h = snake_high[current_index]
                    h_link_width = weight_h.size(1)
                    h_link_height = weight_h.size(0)
                    snake_image[current_width:current_width+h_link_width,current_height:current_height+h_link_height] = weight_h.numpy().transpose()
                    current_width += h_link_width
                    if bias_h is not None:
                        snake_image[current_width,:] = max_value
                        snake_image[current_width,current_height:current_height+h_link_height] = bias_h.numpy()
                        current_width += 1

            snake_images[snake_name] = snake_image

        for snake_name, snake_image in snake_images.items():
            sw.add_figure(rank(f'models/{name}/{snake_name}'), plot_image(snake_image),
                          pl_module.global_step)


    def visualize(self, trainer, pl_module, batch, batch_idx, dataloader_idx, sw, prefix, visualize, log_type, data,
                  level=1000):
        if self._check_to_log(batch_idx, prefix, log_type):
            self.visualize_recursive(
                logger=sw,
                pl_module=pl_module,
                prefix=prefix,
                visualize=visualize,
                batch_idx=batch_idx,
                log_type=log_type,
                data=data,
                level=level
            )

    def visualize_recursive(self, logger, pl_module, batch_idx, prefix, data, visualize, log_type, level):
        if isinstance(data, dict) and level > 0:
            for key, value in data.items():
                new_prefix = f'{prefix}/{key}'
                if key != 'truth' or self._check_truth_to_log(batch_idx, new_prefix, log_type):
                    self.visualize_recursive(
                        logger=logger,
                        pl_module=pl_module,
                        batch_idx=batch_idx,
                        prefix=new_prefix,
                        log_type=log_type,
                        visualize=visualize,
                        data=value,
                        level=level-1
                    )
        elif isinstance(data, (list, tuple)) and level > 0:
            for key, value in enumerate(data):
                self.visualize_recursive(
                    logger=logger,
                    pl_module=pl_module,
                    batch_idx=batch_idx,
                    prefix=f'{prefix}/{key}',
                    log_type=log_type,
                    visualize=visualize,
                    data=value,
                    level=level-1
                )
        else:
            visualize(logger, pl_module, batch_idx, f'{prefix}/{batch_idx}', data)

    def _check_truth_to_log(self, index, key, log_type):
        truth_to_log = self._get_truth_to_log(key, log_type)
        if index in truth_to_log:
            truth_to_log.remove(index)
            return True
        return False

    def _get_truth_to_log(self, key, log_type):
        if key not in self.truth_to_log:
            self.truth_to_log[key] = list(range(self.amounts_to_log[log_type]))
        return self.truth_to_log[key]

    def _check_to_log(self, index, key, log_type):
        truth_to_log = self._get_truth_to_log(key, log_type)
        return index in truth_to_log

    def _get_to_log(self, key, log_type):
        if key not in self.truth_to_log:
            self.truth_to_log[key] = list(range(self.amounts_to_log[log_type]))
        return self.truth_to_log[key]

    def _visualize_wav(self, sw, pl_module, batch_idx, prefix, wav):
        for index, sub_wav in enumerate(wav):
            sw.add_audio(rank(f'{prefix}/{index}'), sub_wav.cpu().numpy(), pl_module.global_step,
                         pl_module.sampling_rate)

    def _visualize_mel(self, sw, pl_module, batch_idx, prefix, mel):
        for index, sub_mel in enumerate(mel):
            sw.add_figure(rank(f'{prefix}/{index}'), plot_spectrogram(sub_mel.cpu().numpy()),
                          pl_module.global_step)

    def _visualize_label(self, sw, pl_module, batch_idx, prefix, label):
        cat_label = self._cat_recursive(label)
        for index, sub_label in enumerate(cat_label):
            sw.add_figure(rank(f'{prefix}/{index}'), plot_categorical(sub_label.squeeze().cpu().numpy()),
                          pl_module.global_step)

    def _visualize_output(self, sw, pl_module, batch_idx, prefix, output):
        sw.add_histogram(rank(prefix), output, pl_module.global_step)

    def _cat_recursive(self, label):
        if isinstance(label, dict):
            label_list = list(label.items())
            label_sorted = list(sorted(label_list, key=lambda pair: pair[0]))
            values = [self._cat_recursive(value) for key, value in label_sorted]
            return torch.cat(values, dim=1)
        else:
            label = label.squeeze()
            if label.dtype not in [torch.int64, torch.int32, torch.int16, torch.int8]:
                label = self._probabilities_to_set(label, dim=1)
            else:
                label = label.squeeze().unsqueeze(1)
            return label

    def _probabilities_to_set(self, probabilities, dim=1):
        amount = 12
        broadcast_shape = (*(probabilities.size()), amount)
        broadcast_dims = tuple(range(len(broadcast_shape)))
        permutation_dims = (broadcast_dims[-1], *broadcast_dims[:-1])
        linspace = torch.broadcast_to(torch.linspace(0 + EPSILON, 1 - EPSILON, amount), broadcast_shape).permute(permutation_dims)
        final_shape = linspace.size()
        cumsum = torch.broadcast_to(torch.cumsum(probabilities, dim=dim), final_shape)
        permutation_dims = (*broadcast_dims[1:-1], broadcast_dims[0], *broadcast_dims[-1:])
        threshold = torch.where(cumsum < linspace, torch.ones_like(cumsum)*2, cumsum)
        argmin = threshold.permute(permutation_dims).argmin(dim=dim)
        return argmin