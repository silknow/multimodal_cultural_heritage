#!/bin/sh
set -e
python -m venv env
. env/bin/activate
which python
which pip
pip install -U pip setuptools wheel
pip install -r requirements.txt
