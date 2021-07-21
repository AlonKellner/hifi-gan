FROM nvcr.io/nvidia/pytorch:21.06-py3

RUN mkdir wavs
WORKDIR app

COPY requirements.txt .
RUN pip install -r requirements.txt

ENV VER=0.0.7

RUN pip install git+https://github.com/AlonKellner/pytorch-summary

ENV PYTHONPATH "${PYTHONPATH}:src"

COPY src src
COPY config config

ENTRYPOINT ["python", "src/speech-distillation/train_autoencoding.py", "--config", "config/config_none.json", "--input_wavs_dir", "/datasets/ljspeech/LJSpeech-1.1/wavs", "--input_training_file", "/datasets/ljspeech/LJSpeech-1.1/training.txt", "--input_validation_file", "/datasets/ljspeech/LJSpeech-1.1/validation.txt", "--checkpoint_path", "/mount"]