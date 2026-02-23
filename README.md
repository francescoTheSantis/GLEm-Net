# GLEm-Net: Unified Framework for Data Reduction with Categorical and Numerical Features

## Overview
This repository contains the code for integrating feature selection in neural networks: [GLEm-Net](https://ieeexplore.ieee.org/document/10386901). The methodology presented here is capable of handling both numerical and categorical variables, enabling efficient feature selection in neural network-based models.

## Features
- **Compatibility**: Works seamlessly with numerical and categorical data.
- **Optimization**: the model is trained while the subset of most informative features is selected. After the training, the model is able to make predictions using only the selected features subset.

## Methodology
The methodology presented here follows a structured approach for feature selection in neural networks. The core steps include:
- **Grouped Lasso Regularization**: Implemented on the first layer of the neural network.
- **Extension to Categorical Features**: Incorporating changes in the loss function to handle categorical variables.
- **Enhanced Feature Selection**: Through further modifications in training and the loss function.

## Usage
For a detailed, step-by-step guide on implementing the methodology, check out example.py in the repository. This file provides a comprehensive walkthrough of the methodology's application within a neural network setting.

## Citation

```
@inproceedings{de2023glem,
  title={GLEm-Net: Unified Framework for Data Reduction with Categorical and Numerical Features},
  author={De Santis, Francesco and Giordano, Danilo and Mellia, Marco and Damilano, Alessia},
  booktitle={2023 IEEE International Conference on Big Data (BigData)},
  pages={4240--4247},
  year={2023},
  organization={IEEE}
}

@article{DESANTIS2026115049,
title = {GLEm-Net: Unified framework for data reduction with categorical and numerical features},
journal = {Knowledge-Based Systems},
volume = {334},
pages = {115049},
year = {2026},
issn = {0950-7051},
doi = {https://doi.org/10.1016/j.knosys.2025.115049},
url = {https://www.sciencedirect.com/science/article/pii/S0950705125020878},
author = {Francesco {De Santis} and Danilo Giordano and Marco Mellia}
}
```
