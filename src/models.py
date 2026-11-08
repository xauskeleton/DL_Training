import torch.nn as nn


def build_model(config: dict) -> nn.Module:
    """
    Khung chuẩn khởi tạo mô hình dựa vào file cấu hình (config).
    Mỗi branch project (P1, P2, ...) sẽ định nghĩa các class mô hình tại đây.
    """
    raise NotImplementedError("Cài đặt hàm build_model() cho mô hình của project ở branch hiện tại.")

