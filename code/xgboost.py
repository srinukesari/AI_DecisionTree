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
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
    print("\nClassification Report:")
    class_report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Indoor', 'Outdoor'], yticklabels=['Indoor', 'Outdoor'])

    with open("xgb_model_results.txt", "w") as file:
        file.write(f"Accuracy: {accuracy:.2f}\n\n")
        file.write(f"Precision: {precision:.2f}\n\n")
        file.write(f"Recall: {recall:.2f}\n\n")
        file.write(f"F1 Score: {f1:.2f}\n\n")
        file.write("Classification Report:\n")
        file.write(class_report + "\n")
        file.write("Confusion Matrix:\n")
        file.write(str(cm) + "\n")
        file.write("Feature Importance:\n")
        for i, importance in enumerate(importances):
            file.write(f"Feature {i}: {importance:.4f}\n")
    joblib.dump(model, 'xgb_best_model.pkl')

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show()
    plt.figure(figsize=(10, 5))
    xgb.plot_importance(model, max_num_features=10)
    plt.title("Feature Importance")
    plt.show()


