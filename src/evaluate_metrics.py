import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve
from collections import Counter

def plot_metrics(history):
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss
    axes[0].plot(epochs, history["train_loss"], label="Train Loss", marker="o")
    axes[0].plot(epochs, history["val_loss"], label="Val Loss", marker="o")
    axes[0].set_title("Loss por época")
    axes[0].set_xlabel("Época")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(epochs, history["train_acc"], label="Train Acc", marker="o")
    axes[1].plot(epochs, history["val_acc"], label="Val Acc", marker="o")
    axes[1].set_title("Accuracy por época")
    axes[1].set_xlabel("Época")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def show_distribution(model, test_loader, test_dataset):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            logits = model(images)
            probs = F.softmax(logits, dim=1)

            _, preds = torch.max(logits, 1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.tolist())
            all_probs.extend(probs[:, 1].cpu().tolist())

    print(f"Predictions distribution: {Counter(all_preds)}")
    print(f"Labels distribution:      {Counter(all_labels)}")
    print(f"Class mapping: {test_dataset.class_to_idx}")

    try:
        auroc = roc_auc_score(all_labels, all_probs)
        print(f"\n[Métrica Forense] Binary AUROC on Known Set: {auroc:.4f}")
    except ValueError:
        print("\n[Warning] AUROC not computed: Only one class present in the batch/dataset.")