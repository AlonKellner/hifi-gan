FROM hammertoe/librosa_ml

RUN mkdir ????
WORKDIR src

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src .


ENTRYPOINT ["python", "????.py", "--????", "????", "--????", "????", "--????", "????"]