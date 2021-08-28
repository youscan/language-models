FROM nvidia/cuda:10.2-cudnn8-runtime-ubuntu18.04
RUN apt-get update &&  apt-get install --fix-missing -y \
    software-properties-common \
    curl \
    lsb-release \
    build-essential \
    libssl-dev \
    libcurl4-openssl-dev \
    python3.6-dev \
    python3.6-distutils

RUN curl https://bootstrap.pypa.io/get-pip.py | python3.6
RUN rm -rf /var/lib/apt/lists/*

RUN mkdir app
WORKDIR /app

RUN python3.6 -m pip install --ignore-installed pycurl

COPY requirements.txt requirements.txt
RUN python3.6 -m pip install -r requirements.txt

COPY src src
COPY setup.py setup.py
RUN python3.6 -m pip install .

COPY run.py run.py
