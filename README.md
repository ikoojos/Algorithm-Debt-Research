# Algorithm Debt Identification using Machine and Deep Learning Models

Welcome to the **AD identification using Machine and Deep Learning Models Repository**! This repository contains several Jupyter notebooks and datasets written using Colab for the identification and classification of AD from differnt TD types. Each notebook demonstrates the use of different models for identifying AD, such as Logistic Regression, Random Forest, Support Vector Machines, and transformer-based embeddings like RoBERTa, ALBERT, and Instructor.

## Repository Structure

```
├── dataset/                # Folder containing the dataset
│   └── liu_datset_processed.csv      # Dataset file
├── notebooks/              # Folder containing the Colab notebooks
│   ├── RoBERTa.ipynb       # Notebook for RoBERTa-based embeddings
│   ├── ALBERT.ipynb        # Notebook for ALBERT-based embeddings
│   ├── Instructor.ipynb    # Notebook for Instructor embeddings
│   ├── LR.ipynb            # Logistic Regression notebook
│   ├── RF.ipynb            # Random Forest notebook
│   ├── SVM.ipynb           # Support Vector Machine notebook
├── README.md               # Documentation file
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

## Contributing
Contributions are welcome! Feel free to create issues, submit pull requests, or suggest improvements.

---

## License
...

---

## Contact
For any questions or issues, please contact:

- **Maintainer**: [ikoojo]
- **Email**: emmanuel.simon@anu.edu.au

---

Thank you for exploring the **AD identification Repository**!
