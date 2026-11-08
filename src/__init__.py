"""
Thư mục src/ chứa toàn bộ mã nguồn cốt lõi của project:
- dataset.py      : Tải dữ liệu, chia train/val/test cố định bằng seed, tạo DataLoader.
- models.py       : Định nghĩa kiến trúc mô hình (MLP, CNN, ResNet, Detector, ViT...).
- engine.py       : Vòng lặp train_one_epoch, evaluate và fit (kèm Early Stopping & TensorBoard).
- sanity_check.py : Kiểm tra loss ban đầu ~ ln(C) và overfit 1 batch nhỏ trước khi train thật.
- utils.py        : Cố định seed, đọc YAML config, EarlyStopping, lưu/nạp checkpoint, tính mean ± std.
- visualize.py    : Vẽ biểu đồ đường Loss/Accuracy và Confusion Matrix.
"""

