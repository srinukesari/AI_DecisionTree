import joblib
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from get_train_and_test_data import get_train_and_test_data 
from evaluate_model import evaluate_model
    
def xgboost():
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = get_train_and_test_data()

    # Train an XGBoost classifier
    model = xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss", use_label_encoder=False)
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    importances = model.feature_importances_
    
    # Evaluate model
    results_boosting = evaluate_model(y_test, y_pred)

    print("comes here")
    print(f"XGBoosting Model Results: {results_boosting}")

    # Print the evaluation metrics
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")

