# Data handling and processing
import pandas as pd
import numpy as np

# Machine learning
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report
from sklearn.svm import SVC
#from sklearn.model_selection import 


# Others
from itertools import product
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Exported symbols
__all__ = [
    "pd", "np", "train_test_split", "Pipeline", 
    "TfidfVectorizer", "CountVectorizer", "StandardScaler", 
    "LogisticRegression", "f1_score", "accuracy_score", 
    "confusion_matrix", "classification_report", "product", "StratifiedKFold"
]
