
# Segmenting breast calcifications using deep learning
<a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-v1.12-red.svg?logo=PyTorch&style=for-the-badge" /></a>
<a href="#"><img src="https://img.shields.io/badge/python-v3.9+-blue.svg?logo=python&style=for-the-badge" /></a>



![](app/assets/20230421111554.png)

A set of different segmentation models to detect calcifications on CBIS-DDSM and KIOS datasets. The FCN network is based on this [paper](https://arxiv.org/abs/2102.00811).


- [Segmenting breast calcifications using deep learning](#segmenting-breast-calcifications-using-deep-learning)
  - [About](#about)
  - [Quick start](#quick-start)
  - [Usage](#usage)
    - [Training](#training)
    - [Web app](#web-app)
  - [Authors](#authors)
  - [License](#license)

## About

Currently, breast cancer is one of the most common types of cancer in the world. The most
commonly used technique to diagnose this disease is mammography. In mammography, the
appearance of calcifications is frequent, which can be benign or malignant and sometimes
difficult for radiologists to detect. In recent years, thanks to deep learning, systems can
be developed that allow for the automatic detection of calcifications. This project provides
an introduction to the current situation of breast cancer, explaining the different types of
mammography and calcifications. The goal of the project is to develop several semantic
segmentation models that allow for the detection of calcifications. To achieve this, a review
of state-of-the-art techniques for medical image segmentation is conducted, and different
architectures are implemented to solve the problem. Additionally, several techniques and
modifications to the architectures are proposed to improve existing results. On the other
hand, the MLOps methodology is implemented to ensure the reproducibility of experiments
and automate development. Finally, a web application is developed to visualize the results
and help the doctor to locate calcifications for further analysis



## Quick start

1. Install Anaconda on your machine.
2. Install the environment to reproduce experiments with: `conda env create -f environment.yml`



## Usage

### Training

To train and reproduce CBIS-DDSM experiments in your computer, you can change `config/train/train_cbis.yaml` if you want and type the following:

`python src/train`

For KIOS dataset:

`python src/train_model.py +dataset=kios +train=train_kios`

You can see training plots and metrics in `dvclive/report.html`.


### Web app

To run the web app you have two options:

1. Run locally by typing `streamlit run ./app/segmentation_app.py`. The app will be running in  http://localhost:8501.
2. Run this [notebook]() in Google Colab.

To run tests:

## Authors

- [@ajf97](https://www.github.com/ajf97)


## License

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.
