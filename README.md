# Algorithm Debt Identification using Machine and Deep Learning Models


Previous studies have shown that Machine Learning (ML) and Deep Learning (DL) models can detect Technical Debt (TD) from source code comments, specifically Self-Admitted Technical Debt (SATD). Despite the importance of ML/DL in software development, no studies focus on the automated detection of new SATD types in ML/DL systems, such as Algorithm Debt (AD). Detecting AD is crucial as it helps identify TD early, facilitating research and learning, and preventing AD-related issues like scalability.

**Aim** Our goal is to investigate the identification performance of various ML/DL models in detecting AD.

**Method** We conducted empirical studies using approaches such as TF-IDF, Count Vectorizer, and Hash Vectorizer with ML/DL classifiers. We used a dataset curated from seven DL frameworks, with comments manually classified into categories: AD, Compatibility, Defect, Design, Documentation, Requirement, and Test Debt. We used embeddings from DL models like ROBERTA and ALBERTv2, as well as large language models (LLMs) such as INSTRUCTOR and VOYAGE AI. We enriched the dataset with AD-related terms and trained various ML/DL classifiers, including Support Vector Machine, Logistic Regression, Random Forest, RoBERTa, and ALBERTv2.

## Repository Structure
The files are organised thus:
```
├── dataset/                          # Folder containing the dataset
│   └── liu_datset_processed.csv      # Dataset file
├── notebooks/                        # Folder containing the Colab notebooks
│   ├── RoBERTa.ipynb                 # Notebook for RoBERTa-based embeddings
│   ├── ALBERT.ipynb                  # Notebook for ALBERT-based embeddings
│   ├── Instructor.ipynb              # Notebook for Instructor embeddings
│   ├── LR.ipynb                      # Logistic Regression notebook
│   ├── RF.ipynb                      # Random Forest notebook
│   ├── SVM.ipynb                     # Support Vector Machine notebook
├── README.md                         # Documentation file
```

---

## Dataset

The dataset required for these notebooks is located in the `dataset` folder. To use the dataset:

1. Download the dataset (`liu_datset_processed.csv`) to your local machine.
2. Update the appropriate file path in the notebook before running it. For example:

   ```python
   dataset_path = "path/to/dataset/liu_datset_processed.csv"
   ```

---

## Notebooks

Each notebook corresponds to a different approach for identifying AD:

1. **RoBERTa.ipynb**: Uses the RoBERTa model to generate embeddings for text data.
2. **ALBERT.ipynb**: Implements the ALBERT model for identifying AD and its embeddings.
3. **Instructor.ipynb**: Uses the Instructor model for embedding comments.
4. **LR.ipynb**: Applies Logistic Regression for AD identification.
5. **RF.ipynb**: Utilises Random Forest for classification tasks.
6. **SVM.ipynb**: Implements a Support Vector Machine (SVM) model for AD identification.


### Scripts

The following Python scripts provide utility functions and preprocessing steps:

1. **utils.py**: Contains helper functions used across the notebooks, including loading datasets and formatting text for model compatibility.
2. **splitting.py**: Handles dataset splitting into training and test sets while ensuring reproducibility and consistency.
3. **preprocessing.py**: Provides functions for cleaning and preprocessing text data, such as tokenization and removal of unnecessary characters.

### Dataset

- **/dataset**: Contains the dataset used for training and testing the models. 
  - Before running the notebooks, download this dataset and set the appropriate file path in your notebook.

---

## Dependencies

### Python Version
The notebooks require **Python 3.8 or later**.

### Required Libraries
Before running any notebook, ensure you have the following libraries installed:

```bash
pip install numpy pandas scikit-learn matplotlib sentence-transformers
```

### Additional Libraries for Transformer Models
To use RoBERTa, ALBERT, or Instructor embeddings, install:

```bash
pip install transformers
```

---

## Usage Instructions

### Step 1: Clone the Repository
Clone the repository to your local machine using:

```bash
git clone https://github.com/your-username/technical-debt-analysis.git
cd technical-debt-analysis
```

### Step 2: Install Dependencies
Install all required dependencies:

```bash
pip install -r requirements.txt
```
(If a `requirements.txt` file is not provided, refer to the dependencies listed above.)

For VOYAGEAI, you need to get a token and replace the part: VOYAGE_API_KEY = "your_api_key"

### Step 3: Open Notebooks
Open the notebooks in Jupyter Notebook, JupyterLab, or Google Colab. For Google Colab:

1. Upload the notebook to your Colab environment.
2. Upload the dataset to your Colab environment or update the file path to match your local setup.

### Step 4: Update Dataset Path
Modify the dataset path in the notebook code to point to the location of `td_dataset.csv`.

### Step 5: Execute Cells
Run the cells in the notebook sequentially to replicate the results or analyze your own data.

---

## Example Workflow
Here is an example of using the `RoBERTa.ipynb` notebook:

1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install transformers sentence-transformers numpy pandas scikit-learn
   ```
3. Open `RoBERTa.ipynb` in your preferred environment.
4. Update the `dataset_path` variable with the correct path to `td_dataset.csv`.
5. Run the cells to preprocess data, generate embeddings, and train the classifier.

---

## Paper:
https://arxiv.org/abs/2408.10529

---

### Registered Report

For further details about the research and methodology, refer to the [Registered Report](https://arxiv.org/pdf/2408.10529).


---

## Contact
For any questions or issues, please contact:

- **Maintainer**: [ikoojo]
- **Email**: emmanuel.simon@anu.edu.au

---

Thank you for exploring the **AD identification Repository**!
