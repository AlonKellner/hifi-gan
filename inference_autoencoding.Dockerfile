FROM nvcr.io/nvidia/pytorch:21.06-py3

RUN mkdir wavs
WORKDIR app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src src
COPY config config

ENV PYTHONPATH "${PYTHONPATH}:src"

ENTRYPOINT ["python", "src/speech-distillation/inference_autoencoding.py", "--input_wavs_dir", "/datasets/ljspeech/LJSpeech-1.1/test", "--output_dir", "/mount/output", "--checkpoint_file", "/mount/g_00015000"]