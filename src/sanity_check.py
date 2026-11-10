from typing import Any
import copy
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def check_initial_loss(
    model: Any,
    loader: DataLoader,
    criterion: nn.Module,
    num_classes: int,
    device: torch.device,
) -> float:
    """
    Sanity Check 1 (Karpathy Recipe):
    Khi mới khởi tạo, xác suất dự đoán mỗi lớp xấp xỉ 1 / num_classes,
    nên Cross-Entropy Loss ban đầu phải xấp xỉ -ln(1/C) = ln(C).
    Ví dụ với 10 lớp (MNIST/CIFAR-10): ln(10) ≈ 2.3026.
    """
    model.eval()
    images, labels = next(iter(loader))
    images, labels = images.to(device), labels.to(device)

    with torch.no_grad():
        logits = model(images)
        if hasattr(model, "compute_loss"):
            init_loss = model.compute_loss(logits, labels).item()
        else:
            init_loss = criterion(logits, labels).item()

    expected_loss = math.log(num_classes)
    print(
        f"[Sanity Check 1] Initial Loss = {init_loss:.4f} "
        f"(Kỳ vọng ln({num_classes}) ≈ {expected_loss:.4f})"
    )
    return init_loss


def check_overfit_single_batch(
    model: Any,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    lr: float = 0.05,
    steps: int = 100,
    num_samples: int = 32,
) -> float:
    """
    Sanity Check 2 (Karpathy Recipe):
    Thử train trên 1 batch nhỏ (ví dụ 32 mẫu) trong vài chục bước.
    Nếu code forward/backward/optimizer đúng, model phải học thuộc lòng (overfit)
    được batch này: loss tiến về ~0 và accuracy đạt 100%.
    Dùng bản sao (deepcopy) của model để không làm ảnh hưởng trọng số gốc.
    """
    temp_model = copy.deepcopy(model)
    if hasattr(temp_model, "to"):
        temp_model.to(device)
    temp_model.train()

    is_tensor_model = hasattr(temp_model, "step_sgd") and hasattr(temp_model, "compute_loss")
    if not is_tensor_model:
        optimizer = torch.optim.SGD(temp_model.parameters(), lr=lr, momentum=0.9)

    images, labels = next(iter(loader))
    images = images[:num_samples].to(device)
    labels = labels[:num_samples].to(device)

    final_loss = 0.0
    final_acc = 0.0
    for _ in range(steps):
        if is_tensor_model:
            logits = temp_model(images)
            loss = temp_model.compute_loss(logits, labels)
            loss.backward()
            temp_model.step_sgd(lr)
        else:
            optimizer.zero_grad()
            logits = temp_model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        final_loss = loss.item()
        preds = logits.argmax(dim=1)
        final_acc = (preds == labels).float().mean().item() * 100.0
        if final_loss < 1e-3 and final_acc == 100.0:
            break

    print(
        f"[Sanity Check 2] Overfit 1 batch ({num_samples} mẫu): "
        f"Final Loss = {final_loss:.4f}, Accuracy = {final_acc:.2f}%"
    )
    return final_loss

