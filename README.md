# **Automated Detection of Algorithm Debt in Deep Learning Frameworks: An Empirical Study**

Previous studies have shown that Machine Learning (ML) and Deep Learning (DL) models can detect Technical Debt (TD) from source code comments, specifically Self-Admitted Technical Debt (SATD). Despite the importance of ML/DL in software development, no studies focus on the automated detection of new SATD types in ML/DL systems, such as Algorithm Debt (AD). Detecting AD is crucial as it helps identify TD early, facilitating research and learning, and preventing AD-related issues like scalability.

Aim: Our goal is to investigate the identification performance of various ML/DL models in detecting AD.

Method: We conducted empirical studies using approaches such as TF-IDF, Count Vectorizer, and Hash Vectorizer with ML/DL classifiers. We used a dataset curated from seven DL frameworks, with comments manually classified into categories: AD, Compatibility, Defect, Design, Documentation, Requirement, and Test Debt. We used embeddings from DL models like ROBERTA and ALBERTv2, as well as large language models (LLMs) such as INSTRUCTOR and VOYAGE AI. We enriched the dataset with AD-related terms and trained various ML/DL classifiers, including Support Vector Machine, Logistic Regression, Random Forest, RoBERTa, and ALBERTv2.

---

## **Installation and Dependencies**
---
### **Python Version**

The project requires **Python 3.8 or later**
```bash
python3 -m venv env
source env/bin/activate
pip install --upgrade pip
```
---


### **Dependencies**
Before running any notebook, ensure you have installed the following libraries:

```bash
pip install numpy pandas scikit-learn matplotlib sentence-transformers
```
Install these required libraries as well:

```bash
pip install -r requirements.txt
```

For additional dependencies for transformer-based models, install:

```bash
!pip install transformers
```

```bash
!pip install huggingface_hub==0.25.2
```

```bash
!pip install --upgrade huggingface_hub
```

For VOYAGEAI, you need to get a token and replace the part: VOYAGE_API_KEY = "your_api_key" from the [VOYAGEAI Homepage](https://dash.voyageai.com/api-keys)

If using CUDA for GPU acceleration, follow [PyTorch Get Started](https://pytorch.org/get-started/locally/) for installation instructions.

---

## **Repository Structure**
Sample python scripts are provided which highlights the various notebooks created in Python.
```
├── dataset/                      # Folder containing the dataset
   └── liu_datset_processed.csv   # Dataset filenotebooks/                       
├── RoBERTa.ipynb                 # Notebook for RoBERTa embeddings
├── ALBERT.ipynb                  # Notebook for ALBERT embeddings
├── Instructor.ipynb              # Notebook for Instructor embeddings
├── LR.ipynb                      # Logistic Regression notebook
├── RF.ipynb                      # Random Forest notebook
├── SVM.ipynb                     # Support Vector Machine notebook                         
├── utils.py                      # Helper functions
├── splitting.py                  # Dataset splitting
├── preprocessing.py              # Text preprocessing
├── requirements.txt              # List of required dependencies
├── README.md                     # Documentation file
```

---


## **Usage**

### **Dataset Setup**

1. Download the dataset (`liu_datset_processed.csv`) to your local machine.
2. Place it in an appropriate folder or upload it to your Google Drive depending on how you want to run it.
3. Locate the cell for filepath, and update the dataset path in your notebook code, e.g.:

   ```python
   file_path = "path/to/dataset/liu_datset_processed.csv"
   ```

### **Downloading the Scripts**
Download the folllowing scripts to your local machine and and uplaod them to the root directory of your folder: lr_tuning, utils.py, evaluate_model.py, and preprocess.py

### **Running the Notebook on Colab**

1. Clone the repository:

   ```bash
   git clone https://github.com/Algorithm-Debt-Research.git
   cd pip install -e .
   
   ```

2. Download the required notebook, and Open any notebook (e.g., `RoBERTa.ipynb` in this case) in Jupyter Notebook, JupyterLab, or Google Colab.
3. Ensure all dependencies are installed.
4. Modify the dataset path as needed and run the cells sequentially.

---


## **Development Setup**

1. Clone the repository:

   ```bash
   git clone https://github.com/Algorithm-Debt-Research.git
   ```

2. Create a virtual environment and activate it:

   ```bash
   python -m venv env
   source env/bin/activate   # On Windows, use `env\Scripts\activate`
   ```

3. Install dependencies for development:

   ```bash
   python -m pip install -e ".[dev]"
   pre-commit install
   ```

4. To run tests:

   ```bash
   pytest
   ```

---

## **Example Workflow on Colab**

Here is an example of running  notebook e.g., `RoBERTa.ipynb` on Google Colab notebook:

1. Clone the repository or download the Colab file.
2. Install dependencies.
   ```bash
   pip install -U sentence-transformers
   ```
4. Open a notebook (e.g., `RoBERTa.ipynb`) in your Jupyter Notebook or upload to Colab.
5. Upload the dataset to Colab and update the `dataset_path` variable with the correct location of `liu_datset_processed.csv`.
6. Change this line to your appropriate working directory in Colab:
   ```bash
   %cd '/content/drive/My Drive/AD Identification using SATD'
   ```
8. Run the cells sequentially to replicate the results.

---

## **Citation**

If you use this repository or its results, please cite the research as follows:

```plaintext
Simon, E.I.O., Hettiarachchi, C., Potanin, A., Suominen, H. and Fard, F., "Automated Detection of Algorithm Debt in Deep Learning Frameworks: An Empirical Study," 2024. [Online]. Available: https://arxiv.org/pdf/2408.10529
```

---

## **Contact**

For questions, feedback, or support, please contact:

- Iko-Ojo on emmanuel.simon@anu.edu.au

---

## **Acknowledgments**

This repository was developed as part of the research on Algorithm Debt (AD) in ML/DL systems. Special thanks to the supervisors of the research.
