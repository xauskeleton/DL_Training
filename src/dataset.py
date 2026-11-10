import os
from typing import Tuple

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def get_dataloaders(
    data_dir: str = "./data/raw",
    batch_size: int = 128,
    val_ratio: float = 0.1,
    seed: int = 42,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Tải và tiền xử lý bộ dữ liệu MNIST cho P1:
    1. Chuẩn hóa ảnh: ToTensor() đưa về [0, 1], Normalize() với mean=0.1307, std=0.3081.
    2. Tách 10% tập train (6,000 ảnh) làm tập Validation cố định bằng seed qua Generator.
    3. Giữ nguyên 10,000 ảnh cho tập Test.
    4. Trả về (train_loader, val_loader, test_loader).
    """
    os.makedirs(data_dir, exist_ok=True)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    # 1. Tải bộ dữ liệu MNIST gốc
    train_val_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform,
    )
    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform,
    )

    # 2. Tách 10% tập train làm tập Validation cố định bằng seed
    total_train = len(train_val_dataset)
    val_size = int(total_train * val_ratio)
    train_size = total_train - val_size

    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(
        train_val_dataset,
        [train_size, val_size],
        generator=generator,
    )

    # 3. Tạo DataLoader
    use_cuda = torch.cuda.is_available()
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_cuda,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_cuda,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_cuda,
    )

    return train_loader, val_loader, test_loader
