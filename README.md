# Multimodal Metadata Assignment for Cultural Heritage Artifacts
This repository contains the code for a Journal article.

## Abstract
In this work, we develop a multimodal classifier for the cultural heritage domain using a late fusion approach and introduce a novel dataset. The main modalities are Image and Text, but we also incorporate tabular data using the labels for the different tasks present in our dataset. The image classifier is based on a ResNet convolutional neural network architecture. The text classifier is based on a multilingual transformer architecture (XML-Roberta). Both are trained as multitask classifiers. Finally, tabular data and late fusion are handled by Gradient Tree Boosting. We present a detailed analysis of the performance of each modality separately and of the final multimodal classifier.

## Data
The data used for the article and this repository is available online at [https://zenodo.org/record/6536232](https://zenodo.org/record/6536232)

## Organization
Each directory corresponds do a different modality classifier. See each
directory for its respective README.

**Note**: Currently the image classifier has not been uploaded.


## Acknowledgments
This work was supported by the Slovenian Research Agency and the European Union's Horizon 2020 research and innovation program under SILKNOW grant agreement No. 769504.


## Other Links

* Project website: [SILKNOW](https://silknow.eu)
* Project github: [SILKNOW GITHUB](https://github.com/silknow)
