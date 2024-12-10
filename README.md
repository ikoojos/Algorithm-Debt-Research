# Algorithm Debt Research

This repository contains codes developed in Google Colab for identifying **Algorithm Debt (AD)** in Deep Learning (DL) frameworks. The aim of this project is to investigate AD identification performance of various Machine Learning (ML)/DL models and embeddings from Self Admitted Technical Debt in source code comments known as SATD through an empirical study.

## Table of Contents
1. [Notebooks](#notebooks)
2. [Data and Embeddings](#data-and-embeddings)
3. [Setup and Usage](#setup-and-usage)

---

## Notebooks

- **Fine_Tuned_RoBERTa_and_Embeddings.ipynb**
   - Contains codes for the experiments using a trained RoBERTa and extracted embeddings from the fine tuned RoBERTa model for AD identification.

- **LR_TFIDF.ipynb**
  - Contains codes for the Logistic Regression-based classifiers for AD identification. This notebook contains the implementation of the LR model.
  
- **SVM_linear_hash_.ipynb**
  - Contains codes for the SVM Linear kernel with the Hashing vectoriser to identify AD.
  
- **Dataset Preprocessing.ipynb**
  - Prepares and preprocesses datasets for training and evaluation.
  - Includes data cleaning, feature engineering, and exploratory analysis for model consistency and performance.

- **RoBERTa_DL.ipynb**
  - Contains codes for the RoBERTa DL model.
  

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

1. Install required libraries, such as `transformers`, for handling DL models.
2. Download the files.
3. Change the path to the dataset to the correct path in your Colab notebook.

---

This repository provides the codes for experiments for the identification AD in ML/DL. Each notebook allows independent experimentation, enabling the testing of various models, embeddings, and AD-related feature engineering methods. 


