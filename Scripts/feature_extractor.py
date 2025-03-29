import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin

class CustomFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, keywords, save_csv=False, csv_path=None):
        self.keywords = keywords
        self.save_csv = save_csv
        self.csv_path = csv_path

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        
        custom_features = [
            {f'contains_{kw}': int(kw in text.lower()) for kw in self.keywords} 
            for text in X
        ]

        
        custom_features_df = pd.DataFrame(custom_features)

        # Save CSV if enabled
        if self.save_csv and self.csv_path:
            combined_df = pd.DataFrame({'Comments': X}).reset_index(drop=True)
            combined_df = pd.concat([combined_df, custom_features_df], axis=1)

            if y is not None:
                combined_df['TDType'] = y.reset_index(drop=True) if isinstance(y, pd.Series) else pd.Series(y).reset_index(drop=True)

            combined_df.to_csv(self.csv_path, index=False)
            print(f"CSV saved to {self.csv_path}")

        # Return sparse matrix of features for the pipeline
        return csr_matrix(custom_features_df.values)

# Define keywords
keywords = ['shape', 'input', 'tensor', 'number', 'matrix']

