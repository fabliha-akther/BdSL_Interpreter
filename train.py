import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
from model import SignLanguageANN, SignLanguageCNN, SignLanguageCNN2
from torch.utils.data import DataLoader, TensorDataset
from data_loader import load_dataset

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load and Prepare Data
digits_path = "../bdsl47_dataset/dataverse_files/Bangla Sign Language Dataset - Sign Digits"
letters_path = "../bdsl47_dataset/dataverse_files/Bangla Sign Language Dataset - Sign Letters"
X1, y1, _ = load_dataset(digits_path)
X2, y2, _ = load_dataset(letters_path)
X = np.concatenate([X1, X2])
y = np.concatenate([y1, y2])

# Encode Labels
classes = np.unique(y)
class_to_idx = {label: idx for idx, label in enumerate(classes)}
y = np.array([class_to_idx[label] for label in y])
output_size = len(classes)

# Normalize + Augment
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

def augment(X):
    aug = []
    for x in X:
        noise = np.random.normal(0, 0.015, size=x.shape)
        scale = np.random.uniform(0.90, 1.10)
        shift = np.random.uniform(-0.03, 0.03, size=x.shape)
        aug.append(x * scale + noise + shift)
    return np.array(aug, dtype=np.float32)

X = augment(X)

# Train/Val/Test Split (80/10/10)
n = len(X)
train_end = int(n * 0.8)
val_end = int(n * 0.9)
X_train, X_val, X_test = X[:train_end], X[train_end:val_end], X[val_end:]
y_train, y_val, y_test = y[:train_end], y[train_end:val_end], y[val_end:]

# Class Weights
counts = np.bincount(y_train)
weights = 1.0 / (np.log1p(counts) + 1e-8)
weights = weights / weights.sum() * output_size
weights = torch.tensor(weights, dtype=torch.float32).to(device)

# DataLoaders
train_loader = DataLoader(TensorDataset(
    torch.tensor(X_train, dtype=torch.float32),
    torch.tensor(y_train, dtype=torch.long)
), batch_size=64, shuffle=True)

val_loader = DataLoader(TensorDataset(
    torch.tensor(X_val, dtype=torch.float32),
    torch.tensor(y_val, dtype=torch.long)
), batch_size=64)

X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test, dtype=torch.long).to(device)

# Models to Train
model_list = [
    ("ANN", SignLanguageANN(output_size=output_size)),
    ("CNN", SignLanguageCNN(output_size=output_size)),
    ("CNN2", SignLanguageCNN2(output_size=output_size))
]

# Training Loop
for name, model in model_list:
    print(f"\n Training {name} model...\n")
    model = model.to(device)
    model_save_name = f"model_{name.lower()}.pt"
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.97)

    train_losses, train_accs, train_f1s = [], [], []
    val_losses, val_accs, val_f1s = [], [], []

    best_val_f1 = 0
    epochs_no_improve = 0
    early_stop_patience = 25
    epochs = 200

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = criterion(model(xb), yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)

        model.eval()
        with torch.no_grad():
            train_X_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
            train_y_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
            train_out = model(train_X_tensor)
            _, train_preds = torch.max(train_out, 1)
            train_correct = (train_preds == train_y_tensor).sum().item()
            train_acc = 100 * train_correct / len(y_train)

            train_f1s_epoch = []
            for cls in range(output_size):
                tp = ((train_preds == cls) & (train_y_tensor == cls)).sum().item()
                fp = ((train_preds == cls) & (train_y_tensor != cls)).sum().item()
                fn = ((train_preds != cls) & (train_y_tensor == cls)).sum().item()
                prec = tp / (tp + fp + 1e-8)
                rec = tp / (tp + fn + 1e-8)
                f1 = 2 * prec * rec / (prec + rec + 1e-8)
                train_f1s_epoch.append(f1)
            train_f1_macro = 100 * np.mean(train_f1s_epoch)

        correct = 0
        preds_all, labels_all = [], []
        val_loss = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                val_loss += criterion(out, yb).item()
                _, preds = torch.max(out, 1)
                correct += (preds == yb).sum().item()
                preds_all.append(preds)
                labels_all.append(yb)

        acc = 100 * correct / len(y_val)
        preds_all = torch.cat(preds_all)
        labels_all = torch.cat(labels_all)

        f1_scores = []
        for cls in range(output_size):
            tp = ((preds_all == cls) & (labels_all == cls)).sum().item()
            fp = ((preds_all == cls) & (labels_all != cls)).sum().item()
            fn = ((preds_all != cls) & (labels_all == cls)).sum().item()
            prec = tp / (tp + fp + 1e-8)
            rec = tp / (tp + fn + 1e-8)
            f1 = 2 * prec * rec / (prec + rec + 1e-8)
            f1_scores.append(f1)
        f1_macro = 100 * np.mean(f1_scores)

        train_losses.append(avg_loss)
        train_accs.append(train_acc)
        train_f1s.append(train_f1_macro)
        val_losses.append(val_loss / len(val_loader))
        val_accs.append(acc)
        val_f1s.append(f1_macro)

        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {acc:.2f}% | F1: {f1_macro:.2f}%")
        scheduler.step()

        if f1_macro > best_val_f1:
            best_val_f1 = f1_macro
            epochs_no_improve = 0
            torch.save(model.state_dict(), model_save_name)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stop_patience:
                print(f" Early stopping at epoch {epoch+1}")
                break

    np.save(f"{name}_train_loss.npy", np.array(train_losses))
    np.save(f"{name}_train_acc.npy", np.array(train_accs))
    np.save(f"{name}_train_f1.npy", np.array(train_f1s))
    np.save(f"{name}_val_loss.npy", np.array(val_losses))
    np.save(f"{name}_val_acc.npy", np.array(val_accs))
    np.save(f"{name}_val_f1.npy", np.array(val_f1s))

    X_all = torch.tensor(X, dtype=torch.float32).to(device)
    y_all = torch.tensor(y, dtype=torch.long).to(device)

    with torch.no_grad():
        out = model(X_all)
        _, preds = torch.max(out, 1)
        test_acc = 100 * (preds == y_all).sum().item() / y_all.size(0)

        f1_scores = []
        for cls in range(output_size):
            tp = ((preds == cls) & (y_all == cls)).sum().item()
            fp = ((preds == cls) & (y_all != cls)).sum().item()
            fn = ((preds != cls) & (y_all == cls)).sum().item()
            prec = tp / (tp + fp + 1e-8)
            rec = tp / (tp + fn + 1e-8)
            f1 = 2 * prec * rec / (prec + rec + 1e-8)
            f1_scores.append(f1)

        test_f1 = 100 * np.mean(f1_scores)

        np.save(f"{name}_true.npy", y_all.cpu().numpy())
        np.save(f"{name}_preds.npy", preds.cpu().numpy())

        result = {
            "model": name,
            "test_accuracy": round(test_acc, 2),
            "test_f1": round(test_f1, 2)
        }
        with open(f"{name}_metrics.json", "w") as f:
            json.dump(result, f, indent=2)

    print(f"Final Evaluation on FULL data: {name}")
    print(f"Accuracy: {test_acc:.2f}% | F1 Score: {test_f1:.2f}%")
    print(f"Saved model: {model_save_name}")
