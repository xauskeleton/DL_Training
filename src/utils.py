import os
import random
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
import yaml


def set_seed(seed: int = 42) -> None:
    """
    Cố định toàn bộ nguồn ngẫu nhiên (Python, NumPy, PyTorch CPU/GPU)
    để kết quả chia tập dữ liệu và khởi tạo trọng số có thể tái lập 100%.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Tự động chọn GPU (CUDA) nếu có, ngược lại dùng CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_config(config_path: str) -> Dict[str, Any]:
    """Đọc file cấu hình YAML trả về dictionary."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def count_parameters(model: nn.Module) -> int:
    """Đếm tổng số tham số được học (requires_grad=True) của mô hình."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_metric: float,
    save_path: str,
) -> None:
    """Lưu trạng thái mô hình và optimizer ra file .pt."""
    Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "best_metric": best_metric,
    }
    torch.save(state, save_path)


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    device: torch.device,
    optimizer: torch.optim.Optimizer = None,
) -> Dict[str, Any]:
    """Nạp lại trọng số tốt nhất từ file checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint


class EarlyStopping:
    """
    Theo dõi chỉ số trên tập Validation (không bao giờ dùng tập Test).
    Dừng sớm khi chỉ số không cải thiện sau `patience` lần đánh giá liên tiếp,
    đồng thời tự động lưu lại checkpoint tốt nhất.
    """

    def __init__(
        self,
        patience: int = 5,
        mode: str = "max",
        min_delta: float = 0.0,
        save_path: str = "outputs/checkpoints/best_model.pt",
    ):
        assert mode in ("max", "min"), "mode chỉ nhận 'max' (accuracy/mAP) hoặc 'min' (loss)"
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.save_path = save_path

        self.best_score = -float("inf") if mode == "max" else float("inf")
        self.best_epoch = 0
        self.counter = 0
        self.early_stop = False

    def step(
        self,
        current_score: float,
        epoch: int,
        model: nn.Module,
        optimizer: torch.optim.Optimizer = None,
    ) -> bool:
        improved = (
            current_score > self.best_score + self.min_delta
            if self.mode == "max"
            else current_score < self.best_score - self.min_delta
        )

        if improved:
            self.best_score = current_score
            self.best_epoch = epoch
            self.counter = 0
            save_checkpoint(model, optimizer, epoch, current_score, self.save_path)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop


def summarize_seeds(values: List[float]) -> str:
    """Tính trung bình và độ lệch chuẩn (mean ± std) qua nhiều seed."""
    arr = np.array(values, dtype=np.float64)
    return f"{arr.mean():.4f} ± {arr.std(ddof=0):.4f}"

