from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from evaluate_model import evaluate_model
from get_train_and_test_data import get_train_and_test_data

def random_forest_model():
    x_train, y_train, x_test, y_test = get_train_and_test_data()
    # Instantiate model with optimized parameters
    model = RandomForestClassifier(
        n_estimators=100,  # Reduce the number of trees (start with a smaller value)
        max_depth=10,  # Limit tree depth (experiment with different values)
        min_samples_split=5,  # Increase min_samples_split (try values like 5 or 10)
        max_features="sqrt",  # Consider using "sqrt" or "log2" for max_features
        random_state=42,
        n_jobs=-1  # Use all available CPU cores
    )

    # Hyperparameter tuning with a reduced search space and cross-validation folds
    param_grid = {
        'n_estimators': [50, 100],  # Reduce the range of n_estimators
        'max_depth': [5, 10],  # Reduce the range of max_depth
        'min_samples_split': [2, 5],  # Reduce the range of min_samples_split
    }

    grid_search = GridSearchCV(
        model,
        param_grid,
        cv=StratifiedKFold(n_splits=3, shuffle=True),  # Reduce the number of CV folds
        scoring='balanced_accuracy',
        verbose=2,
        n_jobs=-1  # Use all available CPU cores
    )

    grid_search.fit(x_train, y_train)
    best_model = grid_search.best_estimator_

    # Train model
    best_model.fit(x_train, y_train)

    # Make predictions
    y_pred = best_model.predict(x_test)

    # Evaluate model
    # results_randomforest = evaluate_model(y_test, y_pred)
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
    feature_importances = best_model.feature_importances_
    with open("random_forest_results.txt", "w") as file:
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

    joblib.dump(best_model, 'random_forest_best_model.pkl')
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
    
    print("comes here")
    print(f"Random Forest Model Results: {results_randomforest}")

if __name__ == "__main__":
    print("started random forest model!!!")
    random_forest_model()