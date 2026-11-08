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
    """Vẽ và lưu biểu đồ đường Loss, Accuracy và Learning Rate Schedule."""
    Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    has_lr = "lr" in history and len(history["lr"]) == len(history["train_loss"])
    num_plots = 3 if has_lr else 2
    fig, axes = plt.subplots(1, num_plots, figsize=(16, 4.5) if has_lr else (12, 4.5))

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

    if has_lr:
        axes[2].plot(epochs, history["lr"], label="Learning Rate", color="tab:green", marker="^", markersize=3)
        axes[2].set_title(f"{title} - LR Schedule")
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("Learning Rate")
        axes[2].set_yscale("log")
        axes[2].legend()
        axes[2].grid(True, linestyle="--", alpha=0.5)

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

