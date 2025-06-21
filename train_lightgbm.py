import numpy as np
import pandas as pd
import json
import lightgbm as lgb
from data_loader import load_dataset
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# Load dataset
X1, y1, _ = load_dataset("../bdsl47_dataset/dataverse_files/Bangla Sign Language Dataset - Sign Digits")
X2, y2, _ = load_dataset("../bdsl47_dataset/dataverse_files/Bangla Sign Language Dataset - Sign Letters")
X = np.concatenate([X1, X2])
y = np.concatenate([y1, y2])

# Normalize
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

# Encode labels
classes = np.unique(y)
class_to_idx = {label: idx for idx, label in enumerate(classes)}
y = np.array([class_to_idx[label] for label in y])

# 80/10/10 split
n = len(X)
train_end = int(0.8 * n)
val_end = int(0.9 * n)

X_train, y_train = X[:train_end], y[:train_end]
X_val, y_val = X[train_end:val_end], y[train_end:val_end]
X_test, y_test = X[val_end:], y[val_end:]

# Train model
model = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)

# Predict on test
y_pred_test = model.predict(X_test)
acc = accuracy_score(y_test, y_pred_test) * 100
f1 = f1_score(y_test, y_pred_test, average='weighted') * 100

# Save metrics
result = {
    "model": "LightGBM",
    "test_accuracy": round(acc, 2),
    "test_f1": round(f1, 2)
}
with open("LightGBM_metrics.json", "w") as f:
    json.dump(result, f, indent=2)

# Save predictions and ground truth
np.save("LightGBM_preds.npy", y_pred_test)
np.save("LightGBM_true.npy", y_test)

print("LightGBM Results")
print(f"Accuracy: {acc:.2f}%")
print(f"F1 Score: {f1:.2f}%")
