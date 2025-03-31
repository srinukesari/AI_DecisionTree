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
    results_randomforest = evaluate_model(y_test, y_pred)
    
    print("comes here")
    print(f"Random Forest Model Results: {results_randomforest}")

if __name__ == "__main__":
    print("started random forest model!!!")
    random_forest_model()