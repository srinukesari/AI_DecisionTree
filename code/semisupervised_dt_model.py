import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from get_train_and_test_data import get_train_and_test_data
from evaluate_model import evaluate_model

def semisupervised_dt_tree():
    x_train, y_train, x_test, y_test = get_train_and_test_data()
    x_labeled, x_unlabeled, y_labeled, y_unlabeled = train_test_split(
        x_train, y_train, test_size=0.5, random_state=42  # Adjust test_size as needed
    )

    # 2. Train initial model on labeled data (optimized)
    print("Training initial model on labeled data...")
    model = DecisionTreeClassifier(max_depth=10, min_samples_leaf=5, min_samples_split=10, random_state=42, splitter="random")
    model.fit(x_labeled, y_labeled)
    print("Initial model training complete.")

    # 3. Iteratively predict on unlabeled data and add to training set (optimized)
    max_iterations = 10  # Limit the number of iterations
    iteration = 0
    while len(x_unlabeled) > 0 and iteration < max_iterations:
        print(f"Iteration {iteration + 1}:")
        print(f"  - Number of unlabeled samples: {len(x_unlabeled)}")

        pseudo_probs = model.predict_proba(x_unlabeled)
        confidence_threshold = 0.9  # Adjust as needed
        confident_indices = np.where(np.max(pseudo_probs, axis=1) >= confidence_threshold)[0]

        print(f"  - Number of confident predictions: {len(confident_indices)}")

        if len(confident_indices) == 0:
            print("  - No confident predictions found. Stopping iterations.")
            break

        pseudo_labels = np.argmax(pseudo_probs[confident_indices], axis=1)
        x_labeled = np.concatenate([x_labeled, x_unlabeled[confident_indices]])
        y_labeled = np.concatenate([y_labeled, pseudo_labels])
        x_unlabeled = np.delete(x_unlabeled, confident_indices, axis=0)

        print("  - Retraining model on updated labeled data...")
        model.fit(x_labeled, y_labeled)
        print("  - Retraining complete.")

        iteration += 1

    print("Semi-supervised learning process finished.")


    # 4. Evaluate on test data
    y_pred = model.predict(x_test)
    results_semisupervised_decisiontree = evaluate_model(y_test, y_pred)

    print(f"Semi-Supervised Decision Tree Model Results: {results_semisupervised_decisiontree}")

if __name__ == "__main__":
    semisupervised_dt_tree()
    print("Semi-supervised Decision Tree model completed successfully !!!")