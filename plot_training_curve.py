import numpy as np
import matplotlib.pyplot as plt
import os

# List of all trained models
model_names = ["ANN", "CNN", "CNN2"]

for model in model_names:
    try:
        # Load saved .npy metrics for the model
        train_loss = np.load(f"{model}_train_loss.npy")
        val_loss   = np.load(f"{model}_val_loss.npy")
        train_acc  = np.load(f"{model}_train_acc.npy")
        val_acc    = np.load(f"{model}_val_acc.npy")
        train_f1   = np.load(f"{model}_train_f1.npy")
        val_f1     = np.load(f"{model}_val_f1.npy")

        epochs = range(1, len(train_loss) + 1)

        # Plotting all 3 curves (Accuracy, Loss, F1)
        fig, axs = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f"{model} Training vs Validation", fontsize=16)

        # Accuracy
        axs[0].plot(epochs, train_acc, label="Train Acc", color="green")
        axs[0].plot(epochs, val_acc, label="Val Acc", color="blue")
        axs[0].set_title("Accuracy")
        axs[0].set_xlabel("Epoch")
        axs[0].set_ylabel("Accuracy (%)")
        axs[0].legend()
        axs[0].grid(True)

        # Loss
        axs[1].plot(epochs, train_loss, label="Train Loss", color="orange")
        axs[1].plot(epochs, val_loss, label="Val Loss", color="red")
        axs[1].set_title("Loss")
        axs[1].set_xlabel("Epoch")
        axs[1].set_ylabel("Loss")
        axs[1].legend()
        axs[1].grid(True)

        # F1 Score
        axs[2].plot(epochs, train_f1, label="Train F1", color="purple")
        axs[2].plot(epochs, val_f1, label="Val F1", color="brown")
        axs[2].set_title("F1 Score")
        axs[2].set_xlabel("Epoch")
        axs[2].set_ylabel("F1 Score (%)")
        axs[2].legend()
        axs[2].grid(True)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        out_file = f"{model}_training_val_curves.png"
        plt.savefig(out_file)
        plt.close()
        print(f" Saved: {out_file}")

    except FileNotFoundError as e:
        print(f"Skipped {model}: missing file ({e.filename})")
