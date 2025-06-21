import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# models
model_names = ["LightGBM", "SVM", "ANN", "CNN", "CNN2"]

# Load metrics
accuracies = []
f1_scores = []

for name in model_names:
    try:
        with open(f"{name}_metrics.json", "r") as f:
            result = json.load(f)
            accuracies.append(result["test_accuracy"])
            f1_scores.append(result["test_f1"])
    except FileNotFoundError:
        print(f"{name}_metrics.json not found")
        accuracies.append(0)
        f1_scores.append(0)

# Accuracy & F1 Score Bar Chart
x = np.arange(len(model_names))
bar_width = 0.35

plt.figure(figsize=(10, 6))
plt.bar(x, accuracies, width=bar_width, label="Accuracy", color="skyblue")
plt.bar(x + bar_width, f1_scores, width=bar_width, label="F1 Score", color="lightgreen")
plt.xticks(x + bar_width / 2, model_names)
plt.ylim(0, 105)
plt.ylabel("Percentage (%)")
plt.title("Model Accuracy and F1 Score Comparison")
plt.legend()

# Add value labels
for i in range(len(model_names)):
    plt.text(i, accuracies[i] + 1, f"{accuracies[i]:.1f}%", ha='center', fontsize=8)
    plt.text(i + bar_width, f1_scores[i] + 1, f"{f1_scores[i]:.1f}%", ha='center', fontsize=8)

plt.tight_layout()
plt.savefig("f1_accuracy_comparison.png")
print("Saved: f1_accuracy_comparison.png")

# Confusion Matrices

for name in model_names:
    try:
        y_pred = np.load(f"{name}_preds.npy")
        y_true = np.load(f"{name}_true.npy")

        cm = confusion_matrix(y_true, y_pred)

        class_labels = np.arange(cm.shape[0])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)

        fig, ax = plt.subplots(figsize=(10, 10))  
        disp.plot(ax=ax, cmap="Blues", colorbar=False, xticks_rotation='vertical')
        plt.title(f"{name} - Confusion Matrix")
        plt.tight_layout()
        plt.savefig(f"{name}_confusion_matrix.png")
        print(f"Saved: {name}_confusion_matrix.png")

    except FileNotFoundError:
        print(f"Skipped confusion matrix for {name} (missing preds/true)")


plt.tight_layout()

