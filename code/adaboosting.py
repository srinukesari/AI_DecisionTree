import numpy as np 
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from evaluate_model import evaluate_model
from get_train_and_test_data import get_train_and_test_data

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
    results_boosting = evaluate_model(y_test, y_pred)

    print("comes here")
    print(f"Ada Boosting Model Results: {results_boosting}")

if __name__ == "__main__":
    print("started adaboosting model!!!")
    adaboosting()
    print("AdaBoosting model completed successfully !!!")
