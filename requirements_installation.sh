#!/usr/bin/env bash
pip install -U pip
pip install -r requirements.txt

pip install git+https://github.com/youscan/ds-shared.git

pip install -e .
