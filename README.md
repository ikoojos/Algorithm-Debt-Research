# **Automated Detection of Algorithm Debt in Deep Learning Frameworks: An Empirical Study**

Previous studies have shown that Machine Learning (ML) and Deep Learning (DL) models can detect Technical Debt (TD) from source code comments, specifically Self-Admitted Technical Debt (SATD). Despite the importance of ML/DL in software development, no studies focus on the automated detection of new SATD types in ML/DL systems, such as Algorithm Debt (AD). Detecting AD is crucial as it helps identify TD early, facilitating research and learning, and preventing AD-related issues like scalability.

Aim: Our goal is to investigate the identification performance of various ML/DL models in detecting AD.

Method: We conducted empirical studies using approaches such as TF-IDF, Count Vectorizer, and Hash Vectorizer with ML/DL classifiers. We used a dataset curated from seven DL frameworks, with comments manually classified into categories: AD, Compatibility, Defect, Design, Documentation, Requirement, and Test Debt. We used embeddings from DL models like ROBERTA and ALBERTv2, as well as large language models (LLMs) such as INSTRUCTOR and VOYAGE AI. We enriched the dataset with AD-related terms and trained various ML/DL classifiers, including Support Vector Machine, Logistic Regression, Random Forest, RoBERTa, and ALBERTv2.

---

## **Repository Structure**

```
├── dataset/                          # Folder containing the dataset
│   └── liu_datset_processed.csv      # Dataset file
├── notebooks/                        # Folder containing the Colab notebooks
│   ├── RoBERTa.ipynb                 # Notebook for RoBERTa embeddings
│   ├── ALBERT.ipynb                  # Notebook for ALBERT embeddings
│   ├── Instructor.ipynb              # Notebook for Instructor embeddings
│   ├── LR.ipynb                      # Logistic Regression notebook
│   ├── RF.ipynb                      # Random Forest notebook
│   ├── SVM.ipynb                     # Support Vector Machine notebook
├── scripts/                          # Utility scripts
│   ├── utils.py                      # Helper functions
│   ├── splitting.py                  # Dataset splitting
│   ├── preprocessing.py              # Text preprocessing
├── requirements.txt                  # List of required dependencies
├── README.md                         # Documentation file
```

---

## **Installation**

### **Python Version**

The project requires **Python 3.8 or later**.

### **Install Dependencies**
Before running any notebook, ensure you have installed the following libraries:

```bash
pip install numpy pandas scikit-learn matplotlib sentence-transformers
```

Install these required libraries as well:

```bash
pip install -r requirements.txt
```
(If a requirements.txt file is not provided, refer to the dependencies listed above.)

For additional dependencies for transformer-based models, install:

```bash
pip install transformers
```

For VOYAGEAI, you need to get a token and replace the part: VOYAGE_API_KEY = "your_api_key" from the [VOYAGEAI Homepage](https://dash.voyageai.com/api-keys)

If using CUDA for GPU acceleration, follow [PyTorch Get Started](https://pytorch.org/get-started/locally/) for installation instructions.

---

## **Usage**

### **Dataset Setup**

1. Download the dataset (`liu_datset_processed.csv`) to your local machine.
2. Place it in an appropriate folder.
3. Update the dataset path in your notebook code, e.g.:

   ```python
   file_path = "path/to/dataset/liu_datset_processed.csv"
   ```

### **Running the Notebooks**

1. Clone the repository:

   ```bash
   git clone https://github.com/your-repository.git
   cd your-repository
   ```

2. Open any notebook (e.g., `RoBERTa.ipynb`) in Jupyter Notebook, JupyterLab, or Google Colab.
3. Ensure all dependencies are installed.
4. Modify the dataset path as needed and run the cells sequentially.

---

## **Scripts**

### **Key Scripts**

1. **`utils.py`**  
   Contains helper functions for loading datasets, formatting text, and other utilities.

2. **`splitting.py`**  
   Handles reproducible dataset splitting for training and testing.

3. **`preprocessing.py`**  
   Provides text preprocessing functions, including tokenization and noise removal.

---

## **Development Setup**

1. Clone the repository:

   ```bash
   git clone https://github.com/your-repository.git
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

## **Example Workflow**

Here is an example of running the `RoBERTa.ipynb` notebook:

1. Clone the repository
2. Install dependencies.
   ```bash
   pip install -U sentence-transformers
   ```
4. Open the notebook (`RoBERTa.ipynb`) in your preferred (Jupyter or Colab).
5. Update the `dataset_path` variable with the location of `liu_datset_processed.csv`.
6. Run the cells sequentially to replicate the results.

---

## **Citation**

If you use this repository or its results, please cite the research as follows:

```plaintext
Simon, E.I.O., Hettiarachchi, C., Potanin, A., Suominen, H. and Fard, F., "Automated Detection of Algorithm Debt in Deep Learning Frameworks: An Empirical Study," 2024. [Online]. Available: https://arxiv.org/pdf/2408.10529
```

---

## **Contact**

For questions, feedback, or support, please contact:

- **Maintainer**: [ikoojo]
- **Email**: emmanuel.simon@anu.edu.au

---

## **Acknowledgments**

This repository was developed as part of the research on Algorithm Debt (AD) in ML/DL systems. Special thanks to the supervisors of the research.
