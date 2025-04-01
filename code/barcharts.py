import re
import matplotlib.pyplot as plt

def extract_xgb_metrics(file_path):
    with open(file_path, 'r') as file:
        text = file.read()
    accuracy_match = re.search(r'Accuracy:\s*([0-9.]+)', text)
    precision_match = re.search(r'Precision:\s*([0-9.]+)', text)
    recall_match = re.search(r'Recall:\s*([0-9.]+)', text)
    f1_score_match = re.search(r'F1 Score:\s*([0-9.]+)', text)
    metrics = {
        'Accuracy': accuracy_match.group(1) if accuracy_match else None,
        'Precision': precision_match.group(1) if precision_match else None,
        'Recall': recall_match.group(1) if recall_match else None,
        'F1-Score': f1_score_match.group(1) if f1_score_match else None
    }
    return metrics

def extract_metrics_from_file(file_path):
    with open(file_path, 'r') as file:
        text = file.read()

    accuracy_match = re.search(r'Accuracy\s+([0-9.]+)', text)
    precision_match = re.search(r'Precision\s+([0-9.]+)', text)
    recall_match = re.search(r'Recall\s+([0-9.]+)', text)
    f1_score_match = re.search(r'F1-Score\s+([0-9.]+)', text)
    metrics = {
        'Accuracy': accuracy_match.group(1) if accuracy_match else None,
        'Precision': precision_match.group(1) if precision_match else None,
        'Recall': recall_match.group(1) if recall_match else None,
        'F1-Score': f1_score_match.group(1) if f1_score_match else None
    }
    return metrics

file_path1 = 'xgb_model_results.txt'
metrics1 = extract_xgb_metrics(file_path1)
file_path2 = 'decision_tree_results.txt'
metrics2 = extract_metrics_from_file(file_path2)
file_path3 = 'random_forest_results.txt'
metrics3 = extract_metrics_from_file(file_path3)
file_path4 = 'semisupervised_decision_tree.txt'
metrics4 = extract_metrics_from_file(file_path4)
# Print the extracted metrics
print("Metrics from xgb_model_results.txt:")
for key, value in metrics1.items():
    print(f'{key}: {value}')
print("\nMetrics from decision_tree_results.txt:")
for key, value in metrics2.items():
    print(f'{key}: {value}')
print("\nMetrics from random_forest_results.txt:")
for key, value in metrics3.items():
    print(f'{key}: {value}')
print("\nMetrics from semisupervised_decision_tree.txt:")
for key, value in metrics4.items():
    print(f'{key}: {value}')

import matplotlib.pyplot as plt

models = ["XGBoost", "Decision Tree", "Random Forest", "Decision Tree (Semi-Supervised)"]
accuracy_values = [
    float(metrics1["Accuracy"])*100,  
    float(metrics2["Accuracy"])*100,  
    float(metrics3["Accuracy"])*100,  
    float(metrics4["Accuracy"])*100
]
colors = 'orange'
plt.figure(figsize=(10, 5))
bars = plt.bar(models, accuracy_values, color=colors)
for bar in bars:
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{bar.get_height()}%', 
             ha='center', va='bottom', fontsize=12, fontweight='bold')
plt.xlabel("Algorithms", fontsize=12)
plt.ylabel("Accuracy (%)", fontsize=12)
plt.title("Comparison of Accuracy Achieved by Each Algorithm", fontsize=14,pad=20)
plt.ylim(60, 90)
plt.xticks(rotation=15) 
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

import matplotlib.pyplot as plt

models = ["XGBoost", "Decision Tree", "Random Forest", "Decision Tree (Semi-Supervised)"]
accuracy_values = [
    float(metrics1["Precision"])*100,  
    float(metrics2["Precision"])*100,  
    float(metrics3["Precision"])*100,  
    float(metrics4["Precision"])*100
]
colors = 'lightblue'
plt.figure(figsize=(10, 5))
bars = plt.bar(models, accuracy_values, color=colors)
for bar in bars:
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{bar.get_height()}%', 
             ha='center', va='bottom', fontsize=12, fontweight='bold')
plt.xlabel("Algorithms", fontsize=12)
plt.ylabel("Precision (%)", fontsize=12)
plt.title("Comparison of Precision Achieved by Each Algorithm", fontsize=14,pad=20)
plt.ylim(60, 90)
plt.xticks(rotation=15) 
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

import matplotlib.pyplot as plt

models = ["XGBoost", "Decision Tree", "Random Forest", "Decision Tree (Semi-Supervised)"]
accuracy_values = [
    float(metrics1["Recall"])*100,  
    float(metrics2["Recall"])*100,  
    float(metrics3["Recall"])*100,  
    float(metrics4["Recall"])*100
]
colors = 'lightpink'
plt.figure(figsize=(10, 5))
bars = plt.bar(models, accuracy_values, color=colors)
for bar in bars:
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{bar.get_height()}%', 
             ha='center', va='bottom', fontsize=12, fontweight='bold')
plt.xlabel("Algorithms", fontsize=12)
plt.ylabel("Recall (%)", fontsize=12)
plt.title("Comparison of Recall Achieved by Each Algorithm", fontsize=14,pad=20)
plt.ylim(60, 90)
plt.xticks(rotation=15) 
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

import matplotlib.pyplot as plt

models = ["XGBoost", "Decision Tree", "Random Forest", "Decision Tree (Semi-Supervised)"]
accuracy_values = [
    float(metrics1["F1-Score"])*100,  
    float(metrics2["F1-Score"])*100,  
    float(metrics3["F1-Score"])*100,  
    float(metrics4["F1-Score"])*100
]
colors = 'lightgreen'
plt.figure(figsize=(10, 5))
bars = plt.bar(models, accuracy_values, color=colors)
for bar in bars:
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{bar.get_height()}%', 
             ha='center', va='bottom', fontsize=12, fontweight='bold')
plt.xlabel("Algorithms", fontsize=12)
plt.ylabel("F1-Score (%)", fontsize=12)
plt.title("Comparison of F1-Score Achieved by Each Algorithm", fontsize=14,pad=20)
plt.ylim(60, 90)
plt.xticks(rotation=15) 
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
