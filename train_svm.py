import numpy as np
import json
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from data_loader import load_dataset

# Load dataset
X1, y1, _ = load_dataset("../bdsl47_dataset/dataverse_files/Bangla Sign Language Dataset - Sign Digits")
X2, y2, _ = load_dataset("../bdsl47_dataset/dataverse_files/Bangla Sign Language Dataset - Sign Letters")
X = np.concatenate([X1, X2])
y = np.concatenate([y1, y2])

# Normalize
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

# SVM
model = SVC(kernel="rbf", C=1.0, gamma="scale")
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation
acc = accuracy_score(y_test, y_pred) * 100
f1 = f1_score(y_test, y_pred, average="weighted") * 100

# Save predictions and true labels
np.save("SVM_preds.npy", y_pred)
np.save("SVM_true.npy", y_test)

# Save JSON summary
metrics = {
    "model": "SVM",
    "test_accuracy": round(acc, 2),
    "test_f1": round(f1, 2)
}
with open("SVM_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

# Output
print("\nSVM Results")
print(f"Accuracy: {acc:.2f}%")
print(f"F1 Score: {f1:.2f}%")
print("Saved: SVM_preds.npy, SVM_true.npy, SVM_metrics.json")
