
import pandas as pd

def preprocess_data(file_path):
    """Preprocesses the data: handles missing values, removes duplicates, and cleans text."""
    data = pd.read_csv(file_path, low_memory=False)
    data['Comments'].fillna('', inplace=True)
    data['TDType'] = data['TDType'].astype(str)
    values_to_remove = ['MULTITHREAD', 'nan', 'removeType']
    replacement_value = 'WITHOUT_CLASSIFICATION'
    data['TDType'].replace(values_to_remove, replacement_value, inplace=True)
    data['Comments'] = data['Comments'].str.replace('content=', '', regex=False)
    data['Comments'] = data['Comments'].str.replace('"', '', regex=False)
    data = data.drop_duplicates(subset=['Comments', 'TDType'])
    return data
    