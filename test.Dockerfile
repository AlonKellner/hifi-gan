FROM nvcr.io/nvidia/pytorch:21.06-py3

RUN mkdir wavs
WORKDIR app

COPY requirements.txt .
RUN pip install -r requirements.txt

ENV VER=0.1.5

RUN pip install git+https://github.com/AlonKellner/pytorch-summary

ENV PYTHONPATH "${PYTHONPATH}:src"

COPY src src
COPY config config

ENTRYPOINT ["python", "src/speech-distillation/test.py"]