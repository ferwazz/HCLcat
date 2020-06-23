FROM python:3

RUN pip install --upgrade pip && \
    pip install \
    black \
    lmfit \
    mutmut \
    numpy \
    pandas \
    pytest==5.0.1 \
    scipy

WORKDIR /workdir
