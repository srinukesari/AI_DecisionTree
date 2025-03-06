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
    results_decisiontree = evaluate_model(y_test, y_pred)
    
    print("comes here")
    print(f"Decision Tree Model Results: {results_decisiontree}")

if __name__ == "__main__":
    decision_tree_model()
    print("Decision Tree model completed successfully !!!")