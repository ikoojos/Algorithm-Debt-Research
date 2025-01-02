
from sklearn.model_selection import train_test_split

def split_data(data):
    """Splits the data into training, validation, and test sets."""
    X = data['Comments']
    y = data['TDType']
    # Split into train and test sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Further split the training set into training and validation sets
    X_train_final, X_val, y_train_final, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    return X_train_final, X_val, X_test, y_train_final, y_val, y_test
    