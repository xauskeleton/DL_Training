import time
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.utils import EarlyStopping, load_checkpoint


def train_one_epoch(
    model: Any,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    use_amp: bool = False,
    scaler: Optional[torch.amp.GradScaler] = None,
) -> Tuple[float, float]:
    """
    Chạy 1 epoch huấn luyện chuẩn.
    - Hỗ trợ cả Bản 1 (Tensor + Autograd qua step_sgd) và Bản 2 (nn.Module qua optimizer.step).
    - Bật model.train() để BatchNorm/Dropout hoạt động ở chế độ huấn luyện.
    - Xoá gradient cũ trước mỗi batch.
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        if hasattr(model, "step_sgd") and hasattr(model, "compute_loss"):
            # Bản 1: Tensor + Autograd thuần
            logits = model(inputs)
            loss = model.compute_loss(logits, targets)
            loss.backward()
            current_lr = optimizer.param_groups[0]["lr"] if optimizer is not None else 0.1
            model.step_sgd(current_lr)
        else:
            # Bản 2: Thư viện chuẩn PyTorch
            optimizer.zero_grad()
            if use_amp and scaler is not None and device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(inputs)
                    loss = criterion(logits, targets)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(inputs)
                loss = criterion(logits, targets)
                loss.backward()
                optimizer.step()

        batch_size = inputs.size(0)
        total_loss += loss.item() * batch_size
        preds = logits.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += batch_size

    avg_loss = total_loss / total
    accuracy = (correct / total) * 100.0
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(
    model: Any,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, List[int], List[int]]:
    """
    Đánh giá mô hình trên tập Validation hoặc Test.
    - Bật model.eval() để cố định BatchNorm/Dropout.
    - Dùng @torch.no_grad() để tắt theo dõi đạo hàm, tiết kiệm VRAM và tăng tốc.
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds: List[int] = []
    all_targets: List[int] = []

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        logits = model(inputs)
        if hasattr(model, "compute_loss"):
            loss = model.compute_loss(logits, targets)
        else:
            loss = criterion(logits, targets)

        batch_size = inputs.size(0)
        total_loss += loss.item() * batch_size
        preds = logits.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += batch_size

        all_preds.extend(preds.cpu().tolist())
        all_targets.extend(targets.cpu().tolist())

    avg_loss = total_loss / total
    accuracy = (correct / total) * 100.0
    return avg_loss, accuracy, all_preds, all_targets


def fit(
    model: Any,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int = 30,
    patience: int = 5,
    checkpoint_path: str = "outputs/checkpoints/best_model.pt",
    log_dir: Optional[str] = None,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    use_amp: bool = False,
) -> Dict[str, List[float]]:
    """
    Vòng lặp huấn luyện đầy đủ có:
    - Early Stopping theo Validation Accuracy (giữ lại checkpoint tốt nhất).
    - Ghi log TensorBoard (train/val loss & accuracy).
    - Tự động nạp lại checkpoint tốt nhất khi kết thúc huấn luyện.
    """
    early_stopping = EarlyStopping(patience=patience, mode="max", save_path=checkpoint_path)
    writer = SummaryWriter(log_dir=log_dir) if log_dir else None
    scaler = torch.amp.GradScaler("cuda") if (use_amp and device.type == "cuda") else None

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "lr": [],
        "epoch_time": [],
    }

    for epoch in range(1, epochs + 1):
        start_time = time.perf_counter()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, use_amp=use_amp, scaler=scaler
        )
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)

        current_lr = optimizer.param_groups[0]["lr"]

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss if scheduler.mode == "min" else val_acc)
            else:
                scheduler.step()

        elapsed = time.perf_counter() - start_time
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)
        history["epoch_time"].append(elapsed)

        if writer is not None:
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Loss/val", val_loss, epoch)
            writer.add_scalar("Accuracy/train", train_acc, epoch)
            writer.add_scalar("Accuracy/val", val_acc, epoch)
            writer.add_scalar("LR", current_lr, epoch)

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] ({elapsed:.2f}s) | "
            f"LR: {current_lr:.6f} | "
            f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%"
        )

        if early_stopping.step(val_acc, epoch, model, optimizer, scheduler=scheduler):
            print(
                f"--> Early Stopping tại epoch {epoch}! "
                f"Best Val Acc = {early_stopping.best_score:.2f}% (epoch {early_stopping.best_epoch})"
            )
            break

    if writer is not None:
        writer.close()

    load_checkpoint(model, checkpoint_path, device, optimizer=optimizer, scheduler=scheduler)
    history["best_epoch"] = [early_stopping.best_epoch]
    history["best_val_acc"] = [early_stopping.best_score]
    return history

