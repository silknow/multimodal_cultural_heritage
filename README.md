# Multimodal Metadata Assignment for Cultural Heritage Artifacts
This repository contains the code for reproducing the results of the journal article *Multimodal Metadata Assignment for Cultural Heritage Artifacts*, https://doi.org/10.1007/s00530-022-01025-2

## Abstract
In this work, we develop a multimodal classifier for the cultural heritage domain using a late fusion approach and introduce a novel dataset. The main modalities are Image and Text, but we also incorporate tabular data using the labels for the different tasks present in our dataset. The image classifier is based on a ResNet convolutional neural network architecture. The text classifier is based on a multilingual transformer architecture (XML-Roberta). Both are trained as multitask classifiers. Finally, tabular data and late fusion are handled by Gradient Tree Boosting. We present a detailed analysis of the performance of each modality separately and of the final multimodal classifier.

## Data
The data used for the article and this repository is available online at [https://zenodo.org/record/6590957](https://zenodo.org/record/6590957)

The code expects a data/ directory to be created in the same directory as this README. This can be modified in the respective scripts.

## Organization
Each directory corresponds do a different modality classifier. See each directory for its respective README.

## Acknowledgments
This work was supported by the Slovenian Research Agency and the European Union's Horizon 2020 research and innovation program under SILKNOW grant agreement No. 769504.

## Additional Links
* Project website: [SILKNOW](https://silknow.eu)
* Project github: [SILKNOW GITHUB](https://github.com/silknow)
