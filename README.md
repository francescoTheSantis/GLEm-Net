# GLEm-Net

## Overview
This repository contains a methodology for integrating a feature selection mechanism into a neural network: [GLEm-Net](https://ieeexplore.ieee.org/document/10386901). The methodology presented here is capable of handling both numerical and categorical variables, enabling efficient feature selection in neural network-based models.

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
```
