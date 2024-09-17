# PAE_Classification

# Improving Prenatal Alcohol Exposure Classification using Data Augmentation in 3D Convolutional Neural Networks

## Overview

This repository contains the code and results from our research on using data augmentation to improve the classification of **Prenatal Alcohol Exposure (PAE)** in medical imaging, specifically using **3D Convolutional Neural Networks (CNNs)**. In this study, we investigated the effects of different rates of data augmentation on model performance and generalizability.

## Authors

- **Kauê T. N. Duarte**  
  Department of Radiology  
  University of Calgary, Calgary, AB, Canada  
  kaue.duarte@ucalgary.ca

- **Anik Das**  
  Department of Engineering  
  University of Calgary, Calgary, AB, Canada  
  anik.das@ucalgary.ca

- **Catherine Lebel**  
  Department of Radiology  
  University of Calgary, Calgary, AB, Canada  
  clebel@ucalgary.ca

- **Mariana P. Bento**  
  Department of Biomedical Engineering  
  University of Calgary, Calgary, AB, Canada  
  mariana.pinheirobent@ucalgary.ca

## Repository Contents

- **`generate_plot`**: Contains the code for plotting all the images used in the study.
- **`subplots.pdf`**: Includes the training curves for all the models tested.

## Installation and Requirements

To run the code and reproduce the results, please ensure you have the following Python packages installed:

- `numpy`
- `pandas`
- `matplotlib`
- `tensorflow` or `pytorch` (depending on the implementation)
- `scikit-learn`

You can install these packages using `pip`:

```bash
pip install numpy pandas matplotlib tensorflow scikit-learn
