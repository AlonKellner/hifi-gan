import warnings
import json
import warnings
from multiprocessing import Pool

from src.speech_distillation.lightning_model import create_datasets, initialize_model_objects

warnings.simplefilter(action='ignore', category=FutureWarning)

import torch


def main():
    config, datasets, model, tb_logger = initialize_model_objects()

    torch.set_printoptions(profile='full')




if __name__ == '__main__':
    main()
