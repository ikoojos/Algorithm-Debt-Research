from sklearn.metrics import confusion_matrix, classification_report

def evaluate_best_model(best_model, best_params, best_score, X_test, y_test):
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
