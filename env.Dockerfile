FROM nvcr.io/nvidia/pytorch:21.06-py3
WORKDIR app

COPY requirements.txt .
RUN pip install -r requirements.txt

ENV VER 1

RUN pip install git+https://github.com/AlonKellner/pytorch-summary

COPY light_requirements.txt .
RUN pip install -r light_requirements.txt