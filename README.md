ocrpostcorrection
================

In 2017 and 2019 a competion on [Post-OCR Text
Correction](https://sites.google.com/view/icdar2019-postcorrectionocr)
was organized. This repository contains the 'working' notebooks for reproducing
the best results results of the competition and possibly improving them. The
code in the notebooks use functionality from the
[ocrpostcorrection](https://github.com/jvdzwaan/ocrpostcorrection) package.

## Install dependencies

``` sh
git clone https://github.com/jvdzwaan/ocrpostcorrection.git
cd ocrpostcorrection
pip install -e .
```

## How to use

This repository contains two sets of notebooks:

-   `local` notebooks to be run locally, e.g., for generating datasets
-   `colab` notebooks to be run on machines with a GPU, e.g., for
    training neural networks

```

    ocrpostcorrection
    ├── LICENSE
    ├── README.md
    ├── colab                                 <- Notebooks to be run on GPU
    │   ├── icdar-task1-hf-evaluation.ipynb   <- Evaluate Huggingface BERT model for task 1
    │   └── icdar-task1-hf-train.ipynb        <- Train Huggingface BERT model for task 1
    └── local                                 <- Notebooks to be run locally
        └── icdar-create-hf-dataset.ipynb     <- Create Huggingface dataset from the icdar data
```