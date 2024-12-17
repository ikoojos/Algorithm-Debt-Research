from itertools import product
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

def hyperparameter_tuning(X_train, y_train, X_val, y_val, param_grid, vectorizer):
    best_score = -1
    best_params = None
    best_model = None

    for C, penalty, max_iter in product(param_grid['C'], param_grid['penalty'], param_grid['max_iter']):
        if penalty == 'elasticnet':
            solver = 'saga'
        else:
            solver = 'lbfgs'

        try:
            pipeline = Pipeline([
                ('vectorizer', vectorizer),  # Use the passed vectorizer
                ('scaler', StandardScaler(with_mean=False)),  # StandardScaler for scaling
                ('clf', LogisticRegression(C=C, penalty=penalty, max_iter=max_iter, solver=solver, random_state=42, class_weight='balanced'))  # Logistic Regression model
            ])

            pipeline.fit(X_train, y_train)
            y_val_pred = pipeline.predict(X_val)
            score = accuracy_score(y_val, y_val_pred)

            if score > best_score:
                best_score = score
                best_params = {'C': C, 'penalty': penalty, 'max_iter': max_iter}
                best_model = pipeline

        except Exception as e:
            print(f"Skipping configuration C={C}, penalty={penalty}, max_iter={max_iter} due to error: {e}")

    return best_model, best_params, best_score
    
    
    
def evaluate_best_model_(best_model, best_params, best_score, X_test, y_test):
    if best_model is not None:
        print(f"Best parameters found: {best_params}")
        print(f"Validation set accuracy: {best_score}")

        # Evaluate the best model on the test set
        y_test_pred = best_model.predict(X_test)
        conf_matrix_test = confusion_matrix(y_test, y_test_pred)
        classification_rep_test = classification_report(y_test, y_test_pred)

        print("\nTest Confusion Matrix:")
        print(conf_matrix_test)
        print("\nTest Classification Report:")
        print(classification_rep_test)
    else:
        print("No valid model found during grid search.")