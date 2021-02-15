FROM python:3.8

RUN pip install \
    black \
    lmfit \
    mutmut \
    numpy \
    pandas \
    pytest==5.0.1 \
    scipy \
    sklearn

WORKDIR /workdir
