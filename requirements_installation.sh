#!/usr/bin/env bash
pip install -U pip
pip install -r requirements.txt
pip install -r requirements.dev.txt

(
  git clone https://github.com/NVIDIA/apex || { echo "Failed to download and install Nvidia apex"; exit 1; }
  cd apex && \
  pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
)

(
  git clone https://github.com/mattilyra/LSH || { echo "Failed to download and install LSH"; exit 1; }
  cd LSH && \
  python setup.py install
)

pip install -e .

pre-commit install
