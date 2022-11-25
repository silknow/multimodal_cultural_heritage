Overview
--------

This software provides python functions for the classification of images, and training and evaluation of classification models. It consists of four main parts:
1. The creation of a dataset (download of the images used in the article),
2. The training of a new classifier (with the parameters mentioned in the article as defaults),
3. The evaluation of an existing classifier,
4. The classification of images using an existing classifier.

Installation and Usage
----------------------

To install clone the repo:

    $ git clone https://github.com/silknow/multimodal_cultural_heritage

Move to the directory of the image classification module image_classifier/ and install the software via:

    $ pip install --upgrade .

Further, the data is expected to be dowloaded from [https://zenodo.org/record/6590957](https://zenodo.org/record/6536232), where the images contained in 'img.tgz.parta[a-h]' have to be saved in a subfolder img/ of the data/ directory, i.e. in data/img/.

Afterwards, the 'main.py' of the image classification module in the directory image_classifier/silknow_image_classification/ can be executed leading to similar results to those reported in the article. Small deviations in the quality metrics are to be expected due to random components in the algorithm.

The result of the software is an output directory per trained model, where the trained model, the evaluation of the model as well as files containing the predictions can be found.
- The files with the predictions are called 'sys_integration_pred_[variable].cvs' and 'sys_integration_pred_[variable]_post.cvs', respectively.
- The first file contains the image-centered predictions and the second file contains the object-centered predictions, where the latter ones have to be passed to the multimodal classifier.
