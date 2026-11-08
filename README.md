# Khung Project Thực Hành Deep Learning

Nhánh `main` chứa **bộ khung (template) chuẩn** dùng chung cho toàn bộ 10 project trong [lo-trinh-project-deep-learning.md](lo-trinh-project-deep-learning.md). Mỗi project được phát triển trên **1 nhánh (branch) riêng** tách ra từ `main`.

## 1. Cấu trúc thư mục chuẩn trên `main`

```text
DL_Training/
├── configs/
│   └── default.yaml          # Khung cấu hình siêu tham số (hyperparameters), seed, đường dẫn
├── src/
│   ├── __init__.py
│   ├── dataset.py            # Tải dữ liệu, lấy tập con & chia train/val/test cố định bằng seed
│   ├── models.py             # Định nghĩa kiến trúc mô hình của từng bài
│   ├── engine.py             # Vòng lặp train_one_epoch(), evaluate(), fit() kèm Early Stopping & TensorBoard
│   ├── sanity_check.py       # Kiểm tra loss ban đầu ~ ln(C) và overfit 1 batch nhỏ về loss ~ 0
│   ├── utils.py              # set_seed(), EarlyStopping, lưu/nạp checkpoint, tính mean ± std
│   └── visualize.py          # Vẽ biểu đồ đường Loss/Accuracy và Confusion Matrix
├── main.py                   # Điểm chạy huấn luyện, sanity check, chạy 3 seed và đánh giá tập Test
├── inference.py                  # Điểm chạy suy luận (inference) trên dữ liệu thực tế ngoài tập mẫu
├── requirements.txt          # Các thư viện cần thiết
├── .gitignore                # Bỏ qua dữ liệu thô, trọng số .pt/.pth, log và file tạm của IDE
└── lo-trinh-project-deep-learning.md
```

## 2. Quy trình tạo branch cho mỗi bài (Project)

Mỗi khi bắt đầu một bài mới (ví dụ `p1-mlp`, `p2-cnn`, ...), luôn tạo nhánh mới từ `main` để kế thừa bộ khung chuẩn:

```bash
# Đảm bảo đang đứng ở nhánh main
git checkout main

# Tạo nhánh mới cho bài học tương ứng
git checkout -b p1-mlp
```

---

## 3. Mẫu Báo Cáo Chuẩn Cho Mỗi Branch Project

*(Khi sang branch của từng bài, điền đầy đủ các mục dưới đây)*

### 3.1. Dữ liệu
- **Bộ dữ liệu gốc:**
- **Cách lấy tập con & chia Train / Val / Test (cố định bằng seed):**
- **Số lớp & số mẫu mỗi lớp:**

### 3.2. Dự đoán trước khi chạy (Bắt buộc ghi trước khi chạy thí nghiệm)
| Thí nghiệm | Kết quả kỳ vọng | Lý do dự đoán | Kết quả thực tế | Giải thích chỗ lệch |
| :--- | :--- | :--- | :--- | :--- |
| Thí nghiệm 1 | | | | |

### 3.3. Checklist Sanity Check trước khi Train thật
- [ ] Loss ban đầu khi mới khởi tạo $\approx \ln(\text{số lớp})$ (với 10 lớp $\approx 2.3026$).
- [ ] Overfit 1 batch nhỏ (32 mẫu) đạt Loss $\approx 0$ và Accuracy $= 100\%$.
- [ ] Đã kiểm tra trực tiếp vài mẫu và nhãn sau `DataLoader`.
- [ ] Đã kiểm tra `model.train()` / `model.eval()` và `optimizer.zero_grad()`.

### 3.4. Bảng kết quả thực nghiệm (`mean ± std` qua 3 seed)
| Cấu hình / Mô hình | Val Accuracy (%) | Test Accuracy (%) | Epoch dừng (Early Stopping) | Thời gian / Epoch (s) |
| :--- | :--- | :--- | :--- | :--- |
| | | | | |

### 3.5. Kết quả Inference (`infer.py`) & Phân tích lỗi
- **Dữ liệu ngoài dùng để test:**
- **Thời gian suy luận trung bình (ms/mẫu):**
- **Các ca dự đoán sai và nguyên nhân:**

### 3.6. Những điều còn chưa hiểu & Trả lời câu hỏi kiểm tra
- ...
