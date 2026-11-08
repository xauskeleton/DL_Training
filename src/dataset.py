from typing import Tuple
import torch
from torch.utils.data import DataLoader


def get_dataloaders(
    data_dir: str = "./data/raw",
    batch_size: int = 128,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Khung chuẩn tạo DataLoader cho từng project:
    1. Tải bộ dữ liệu gốc (hoặc lấy tập con nhỏ cố định bằng seed).
    2. Tách tập Validation (ví dụ 10% từ tập train) bằng torch.Generator().manual_seed(seed).
    3. Trả về (train_loader, val_loader, test_loader).
    """
    raise NotImplementedError("Cài đặt hàm get_dataloaders() cho dữ liệu của project ở branch hiện tại.")

