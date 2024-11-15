# Algorithm Debt Research

This repository contains Jupyter notebooks developed in Google Colab for identifying **Algorithm Debt (AD)** in Deep Learning (DL) frameworks. The aim of this project is to investigate AD identification performance of various
Machine Learning (ML)/DL models and embedding from source code comments known as SATD through an empirical study.

To access the files in Google Drive, use this [link](https://drive.google.com/drive/folders/1swdyQlbX3OY6wG2RPfXP9kjIS44oQAz9) to access the project in Google Drive.


## Table of Contents
1. [Notebooks](#notebooks)
2. [Data and Embeddings](#data-and-embeddings)
3. [Setup and Usage](#setup-and-usage)

---

## Notebooks

- **RoBERTa.ipynb**
 - Contains codes for the experiments using RoBERTa and RoBERTa embeddings for AD identification [link](https://colab.research.google.com/drive/1P6lG_EnCNbSxsHSaU22bE5s0uqNe0Pqt#scrollTo=pKh3sNpN7-Ui)

- **SVM.ipynb**
  - Contains codes for the Support Vector Machine (SVM)-based classifiers for text classification tasks. This notebook contains the implementation of the various kernels.
  
- **LR2.ipynb**
  - Contains codes for the Logistic Regression (LR) models to identify AD from source code.
  
- **Dataset Preprocessing.ipynb**
  - Prepares and preprocesses datasets for training and evaluation.
  - Includes data cleaning, feature engineering, and exploratory analysis for model consistency and performance.

- **Indicators and RoBERTa.ipynb**
  - Contains codes for the RoBERTa embeddings with AD-indicative keywords to improve classification accuracy.
  - Explores feature engineering methods by incorporating AD indicators.

- **Albert_Experiments.ipynb**
  - Contains codes for the implementation of the ALBERT model for AD classification.

- **INSTRUCTOR MODEL_LR.ipynb**
  - Contains codes for the for INSTRUCTOR embeddings combined with Logistic Regression for AD identification.

- **Voyage AI.ipynb**
  - Contains codes for the for Voyage AI embeddings combined with Logistic Regression for AD identification.
---

## Data and Embeddings

- **Indicator_roberta_embedding.csv**
  - A CSV file with AD indicators and RoBERTa embeddings, preprocessed for use in experiments.

- **Processed_liu.csv**
  - A cleaned and structured version of the Liu et al. (2020) dataset, for model training.

- **Albert_Embeddings.csv**
  - ALBERT model-generated embeddings for use in ML/DL experiments on AD.

---

## Setup and Usage

To run the notebooks:

1. Open each file in Google Colab via the provided link or by uploading it directly.
2. Install required libraries, such as `transformers`, for handling RoBERTa and ALBERT models.
3. Follow each notebook’s instructions for data preprocessing, model training, and result analysis.

---

This repository provides the codes for experiments for the identification AD in ML/DL. Each notebook allows independent experimentation, enabling the testing of various models, embeddings, and AD-related feature engineering methods. 


