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
    best_params = model.get_params()
    feature_importances = model.feature_importances_
    with open("semisupervised_decision_tree.txt", "w") as file:
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
    joblib.dump(model, 'semi_supervised_decision_tree_model.pkl')

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
    semisupervised_dt_tree()
    print("Semi-supervised Decision Tree model completed successfully !!!")