import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

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
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
) -> None:
    """Lưu trạng thái mô hình, optimizer và scheduler ra file .pt."""
    Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "best_metric": best_metric,
    }
    torch.save(state, save_path)


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
) -> Dict[str, Any]:
    """Nạp lại trạng thái mô hình, optimizer và scheduler từ file checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
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
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
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
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_metric=current_score,
                save_path=self.save_path,
                scheduler=scheduler,
            )
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: Dict[str, Any],
    epochs: int,
) -> Optional[torch.optim.lr_scheduler.LRScheduler]:
    """
    Khởi tạo bộ điều chỉnh tốc độ học (Learning Rate Scheduler) dựa trên cấu hình:
    - 'none': Giữ LR cố định trong suốt quá trình huấn luyện.
    - 'step': Giảm LR sau mỗi `step_size` epoch theo tỷ lệ `gamma`.
    - 'multistep': Giảm LR tại các mốc epoch cụ thể `milestones` (hữu ích cho P3 Faster R-CNN).
    - 'cosine': Cosine Annealing giảm mượt mà về `min_lr`.
    - 'warmup_cosine': Tăng tuyến tính trong vài epoch đầu rồi giảm theo Cosine (chuẩn hiện đại).
    - 'plateau': Giảm LR khi chỉ số Validation chững lại (ReduceLROnPlateau).
    """
    sched_cfg = config.get("train", {}).get("scheduler", {})
    name = sched_cfg.get("name", "none")
    if name is None:
        name = "none"
    name = str(name).lower()

    if name in ("none", "fixed"):
        return None

    if name == "step":
        step_size = sched_cfg.get("step_size", 10)
        gamma = sched_cfg.get("gamma", 0.1)
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    if name == "multistep":
        milestones = sched_cfg.get("milestones", [int(epochs * 0.67)])
        gamma = sched_cfg.get("gamma", 0.1)
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

    if name == "cosine":
        min_lr = float(sched_cfg.get("min_lr", 1e-5))
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=min_lr)

    if name in ("warmup_cosine", "cosine_warmup"):
        warmup_epochs = sched_cfg.get("warmup_epochs", 5)
        min_lr = float(sched_cfg.get("min_lr", 1e-5))
        warmup_epochs = min(warmup_epochs, max(1, epochs - 1))
        warmup_sched = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs
        )
        cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, epochs - warmup_epochs), eta_min=min_lr
        )
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_sched, cosine_sched],
            milestones=[warmup_epochs],
        )

    if name == "plateau":
        mode = sched_cfg.get("mode", "max")
        factor = sched_cfg.get("factor", 0.1)
        patience = sched_cfg.get("patience", 3)
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode=mode, factor=factor, patience=patience
        )

    raise ValueError(
        f"Scheduler '{name}' không được hỗ trợ. Các tùy chọn: none, step, multistep, cosine, warmup_cosine, plateau"
    )


def build_optimizer(
    model: nn.Module,
    config: Dict[str, Any],
) -> torch.optim.Optimizer:
    """
    Khởi tạo bộ tối ưu hóa (Optimizer) từ cấu hình:
    - 'sgd': Stochastic Gradient Descent (hỗ trợ momentum, nesterov, weight_decay).
    - 'adam': Adaptive Moment Estimation (hỗ trợ lr, betas, eps, weight_decay).
    - 'adamw': Adam with Decoupled Weight Decay (chuẩn hiện đại cho CNN/ViT/LLM).
    - 'rmsprop': Root Mean Square Propagation (hỗ trợ lr, alpha, momentum, weight_decay).
    """
    train_cfg = config.get("train", {})
    opt_cfg = train_cfg.get("optimizer", {})

    if isinstance(opt_cfg, str):
        name = opt_cfg.lower()
        opt_cfg = {}
    else:
        name = opt_cfg.get("name", "sgd").lower()

    lr = float(opt_cfg.get("lr", train_cfg.get("lr", 0.1)))
    weight_decay = float(opt_cfg.get("weight_decay", train_cfg.get("weight_decay", 0.0)))
    params = [p for p in model.parameters() if p.requires_grad]

    if name == "sgd":
        momentum = float(opt_cfg.get("momentum", train_cfg.get("momentum", 0.9)))
        nesterov = bool(opt_cfg.get("nesterov", False))
        return torch.optim.SGD(
            params,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov,
        )

    if name == "adam":
        betas = tuple(opt_cfg.get("betas", (0.9, 0.999)))
        eps = float(opt_cfg.get("eps", 1e-8))
        return torch.optim.Adam(
            params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )

    if name == "adamw":
        betas = tuple(opt_cfg.get("betas", (0.9, 0.999)))
        eps = float(opt_cfg.get("eps", 1e-8))
        return torch.optim.AdamW(
            params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )

    if name == "rmsprop":
        alpha = float(opt_cfg.get("alpha", 0.99))
        momentum = float(opt_cfg.get("momentum", 0.0))
        return torch.optim.RMSprop(
            params,
            lr=lr,
            alpha=alpha,
            momentum=momentum,
            weight_decay=weight_decay,
        )

    raise ValueError(
        f"Optimizer '{name}' không được hỗ trợ. Các tùy chọn: sgd, adam, adamw, rmsprop"
    )


def summarize_seeds(values: List[float]) -> str:
    """Tính trung bình và độ lệch chuẩn (mean ± std) qua nhiều seed."""
    arr = np.array(values, dtype=np.float64)
    return f"{arr.mean():.4f} ± {arr.std(ddof=0):.4f}"

