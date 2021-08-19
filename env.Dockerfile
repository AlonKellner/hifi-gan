FROM nvcr.io/nvidia/pytorch:21.06-py3
WORKDIR app

COPY requirements.txt .
RUN pip install -r requirements.txt

RUN pip install git+https://github.com/AlonKellner/pytorch-summary