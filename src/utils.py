import glob
import os
import matplotlib
import torch
from torch.nn.utils import weight_norm

matplotlib.use("Agg")
import matplotlib.pylab as plt
from matplotlib import colors

all_colors = [*colors.BASE_COLORS.values(), *colors.TABLEAU_COLORS.values(), *colors.CSS4_COLORS.values()]
all_colors = all_colors * 10
all_colors = all_colors[:300]


def plot_spectrogram(spectrogram):
    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower",
                   interpolation='none')
    plt.colorbar(im, ax=ax)

    fig.canvas.draw()
    plt.close()

    return fig


def plot_image(image):
    fig, ax = plt.subplots()
    im = ax.imshow(image, aspect="auto", interpolation='none')
    plt.colorbar(im, ax=ax)

    fig.canvas.draw()
    plt.close()

    return fig


def plot_categorical(categorical):
    cmap = colors.ListedColormap(all_colors)
    bounds = [i - 0.5 for i in range(len(all_colors))]
    norm = colors.BoundaryNorm(bounds, cmap.N)
    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(categorical, aspect="auto", origin="lower",
                   interpolation='none', cmap=cmap, norm=norm)
    plt.colorbar(im, ax=ax)

    fig.canvas.draw()
    plt.close()

    return fig


def plot_matrix(matrix):
    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.matshow(matrix, aspect="auto",
                    interpolation='none')
    plt.colorbar(im, ax=ax)

    fig.canvas.draw()
    plt.close()

    return fig


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def apply_weight_norm(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        weight_norm(m)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict


def save_checkpoint(filepath, obj):
    print("Saving checkpoint to {}".format(filepath))
    torch.save(obj, filepath)
    print("Complete.")


def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '????????')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return None
    return sorted(cp_list)[-1]
