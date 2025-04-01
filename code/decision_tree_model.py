import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from get_train_and_test_data import get_train_and_test_data
from evaluate_model import evaluate_model

def decision_tree_model():
    x_train, y_train, x_test, y_test = get_train_and_test_data()
    model = DecisionTreeClassifier()

    param_grid = {
        'max_depth': [3, 5, 7, 10],
        'min_samples_split': [2, 5, 10, 20],
        'criterion': ['gini', 'entropy'],
    }

    # Check if X_train and y_train are empty
    if len(x_train) == 0 or len(y_train) == 0:
        raise ValueError("X_train or y_train is empty. Check your data loading.")

    # Check if the number of images and labels match
    if len(x_train) != len(y_train):
        raise ValueError("Number of images and labels do not match. Check your data loading.")


    grid_search = GridSearchCV(model, param_grid, cv=StratifiedKFold(n_splits=5, shuffle=True), scoring='balanced_accuracy', verbose=2, n_jobs=-1)

    try:
        grid_search.fit(x_train, y_train)
    except IndexError as e:
        print("IndexError occurred during GridSearchCV.fit:")
        print("Error message:", e)
        print("X_train shape:", x_train.shape)
        print("y_train shape:", y_train.shape)
        print("param_grid:", param_grid)
        # Add any other relevant debugging information here
        raise  # Re-raise the exception to stop execution


    best_model = grid_search.best_estimator_


    # Train model
    best_model.fit(x_train, y_train)

    # Make predictions
    y_pred = best_model.predict(x_test)

    # Evaluate model
    # results_decisiontree = evaluate_model(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred,output_dict=True)
    report_df = pd.DataFrame(class_report).transpose()
    results = {
        "Metric": ["Accuracy", "Precision", "Recall", "F1-Score"],
        "Score": [accuracy, precision, recall, f1]
    }
    df_metrics = pd.DataFrame(results)
    feature_importances = best_model.feature_importances_

    with open("decision_tree_results.txt", "w") as file:
        file.write("\nMetrics:\n")
        df_metrics.to_string(file, index=False)
        file.write("\nClassification Report:\n")
        file.write("\n")
        file.write(report_df.to_string(index=True))
        file.write("Confusion Matrix:\n")
        file.write(str(conf_matrix) + "\n")
        file.write("\nBest Hyperparameters\n")
        file.write(str(best_params) + "\n")
        file.write("\nFeature Importances:\n")
        for i, importance in enumerate(feature_importances):
            file.write(f"Feature {i}: {importance:.4f}\n")

    joblib.dump(best_model, 'decision_tree_best_model.pkl')
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1-score: {f1}")
    print("Confusion Matrix:")
    print(conf_matrix)
    plot_tree(best_model)
    plt.figure(figsize=(15, 10))
    feature_names = [f"Feature {i+1}" for i in range(X_train.shape[1])]
    plot_tree(best_model, filled=True, feature_names=feature_names, class_names=["Class1", "Class2"])
    plt.title("Decision Tree Visualization")
    plt.show()
    plt.figure(figsize=(6, 5))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Class1', 'Class2'], yticklabels=['Class1', 'Class2'])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show()
    # Class 1 for indoor
    # Class 2 for outdoor
    
    print("comes here")
    print(f"Decision Tree Model Results: {results_decisiontree}")

if __name__ == "__main__":
    decision_tree_model()
    print("Decision Tree model completed successfully !!!")