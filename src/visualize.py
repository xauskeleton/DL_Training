import os
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_learning_curves(
    history: Dict[str, List[float]],
    save_path: str = "outputs/figures/learning_curves.png",
    title: str = "Train vs Validation Curves",
) -> None:
    """Vẽ và lưu biểu đồ đường Loss và Accuracy của tập Train và Validation."""
    Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(epochs, history["train_loss"], label="Train Loss", marker="o", markersize=3)
    axes[0].plot(epochs, history["val_loss"], label="Val Loss", marker="s", markersize=3)
    axes[0].set_title(f"{title} - Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.5)

    axes[1].plot(epochs, history["train_acc"], label="Train Acc (%)", marker="o", markersize=3)
    axes[1].plot(epochs, history["val_acc"], label="Val Acc (%)", marker="s", markersize=3)
    axes[1].set_title(f"{title} - Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(
    y_true: List[int],
    y_pred: List[int],
    class_names: Optional[List[str]] = None,
    save_path: str = "outputs/figures/confusion_matrix.png",
    title: str = "Confusion Matrix on Test Set",
) -> None:
    """Vẽ và lưu ma trận nhầm lẫn (Confusion Matrix) trên tập Test."""
    Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names if class_names else "auto",
        yticklabels=class_names if class_names else "auto",
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("Nhãn dự đoán (Predicted)")
    ax.set_ylabel("Nhãn thực tế (True)")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)

