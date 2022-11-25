Overview
--------

This directory contains the code necessary to run the tabular and multimodal classifiers.

Usage
-----
    ./install.sh
    source env/bin/activate
    ./run_tabular.sh
    ./run_multimodal.sh

Installation
------------

Data: https://zenodo.org/record/6590957

For the tabular classifier, use `dataset.tsv`, which is expected to be in the base directory of this repository at data/dataset/dataset.tsv.
This corresponds to the base dataset.

For the multimodal classifier, use `dataset_multimodal.tsv`, which is expected to be in the base directory of this repository at data/multimodal/dataset_multimodal.tsv.
This corresponds to the validation and test splits of the original dataset with predictions already made corresponding to the exact run/values reported in the article.

Notes
-----

The hyperparameter search space is defined within `gbcls.py`.
