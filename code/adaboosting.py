import numpy as np 
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from evaluate_model import evaluate_model
from get_train_and_test_data import get_train_and_test_data
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

def adaboosting():
    x_train, y_train, x_test, y_test = get_train_and_test_data()

    base_estimator = DecisionTreeClassifier(max_depth=1)

    # 2. Instantiate AdaBoostClassifier with optimized parameters:
    model = AdaBoostClassifier(
        estimator=base_estimator,
        n_estimators=50,
        learning_rate=1.0,
        random_state=42
    )

    param_grid = {
        'estimator__criterion': ['gini', 'entropy'],  # Access base estimator's criterion
        'n_estimators': [50, 100],  # Example: Tune n_estimators for AdaBoost
        'learning_rate': [0.1, 1.0],  # Example: Tune learning_rate
    }

    # 3. Hyperparameter tuning
    grid_search = GridSearchCV(
        model,
        param_grid,
        cv=StratifiedKFold(n_splits=3, shuffle=True),
        scoring='balanced_accuracy',
        verbose=2,  # You can adjust verbose level here if needed
        n_jobs=-1
    )
    grid_search.fit(x_train, y_train)  # Fit the model

    best_model = grid_search.best_estimator_

    # Train model
    best_model.fit(x_train, y_train)

    # Make predictions
    y_pred = best_model.predict(x_test)

    # Evaluate model
    # results_boosting = evaluate_model(y_test, y_pred)

    print("comes here")
    print(f"Ada Boosting Model Results: {results_boosting}")
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    class_report = classification_report(y_test, y_pred,output_dict=True)
    report_df = pd.DataFrame(class_report).transpose()
    results = {
        "Metric": ["Accuracy", "Precision", "Recall", "F1-Score"],
        "Score": [accuracy, precision, recall, f1]
    }
    df_metrics = pd.DataFrame(results)
    conf_matrix = confusion_matrix(y_test, y_pred)
    best_params = best_model.get_params()
    feature_importances = best_model.feature_importances_
    with open("adaboost.txt", "w") as file:
        file.write("\nMetrics:\n")
        df_metrics.to_string(file, index=False)
        file.write("\nClassification Report:\n")
        file.write("\n")
        file.write(report_df.to_string(index=True))
        file.write("Confusion Matrix:\n")
        file.write(str(conf_matrix) + "\n")
        file.write("\nBest Hyperparameters:\n")
        file.write(str(best_params) + "\n")
        file.write("\nFeature Importances:\n")
        for i, importance in enumerate(feature_importances):
            file.write(f"Feature {i}: {importance:.4f}\n")
    joblib.dump(model, 'adaboost_model.pkl')

    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1-score: {f1}")
    print("Confusion Matrix:")
    print(conf_matrix)
    plt.figure(figsize=(6, 5))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix Heatmap")
    plt.show()
# Class 0 for indoor
# Class 1 for outdoor


if __name__ == "__main__":
    print("started adaboosting model!!!")
    adaboosting()
    print("AdaBoosting model completed successfully !!!")
