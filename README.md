# Project 1: Build Lại MLP Bằng PyTorch (MNIST)

Branch `p1-mlp` triển khai toàn bộ bài toán **P1 — Xây dựng lại Multi-Layer Perceptron (MLP)** theo lộ trình thực hành Deep Learning, nhằm hiểu sâu cơ chế hoạt động của PyTorch từ mức độ Tensor thuần túy đến thư viện cấp cao `torch.nn`.

---

## 1. Hướng Dẫn Chạy Toàn Bộ Thí Nghiệm Theo Bộ Khung Chuẩn

Mọi thí nghiệm đều được điều phối qua điểm chạy duy nhất [`main.py`](main.py) và cấu hình qua các file YAML trong thư mục [`configs/`](configs/):

### Thí nghiệm 1: Huấn luyện Bản 2 (Thư viện chuẩn `torch.nn`)
Huấn luyện MLP dùng `nn.Module`, `nn.Linear`, `build_optimizer(SGD)` mặc định qua 3 seed (`42, 43, 44`) kèm Sanity Check và Early Stopping:
```bash
python main.py --config configs/default.yaml
```

### Thí nghiệm 2: Huấn luyện Bản 1 (Tensor thuần + Autograd)
Chạy bản MLP chỉ dùng ma trận tensor, hàm Cross-Entropy Log-Sum-Exp tự viết và cập nhật 1 dòng SGD thuần:
```bash
python main.py --config configs/p1_tensor.yaml
```

### Thí nghiệm 3: Khảo sát 3 cách khởi tạo trọng số (Ablation Init)
- Khởi tạo mặc định Kaiming: `python main.py --config configs/default.yaml`
- Khởi tạo toàn số 0: `python main.py --config configs/p1_zeros.yaml`
- Khởi tạo ngẫu nhiên quá lớn: `python main.py --config configs/p1_large.yaml`

### Thí nghiệm 4: Chạy suy luận (Inference) trên ảnh chữ số thực tế
Nhận một ảnh chữ số bất kỳ (tự chụp hoặc viết tay), tự động đảo màu nền nếu cần, tiền xử lý về 28×28 và dự đoán:
```bash
python inference.py --input path/to/digit_image.png --checkpoint outputs/checkpoints/best_model_seed_42.pt
```

---

## 2. Báo Cáo Thực Nghiệm P1

### 2.1. Dữ liệu
- **Bộ dữ liệu gốc:** MNIST (Modified National Institute of Standards and Technology).
- **Phân chia tập dữ liệu:**
  - Tổng số ảnh: 70,000 ảnh chữ số xám kích thước $28 \times 28$ pixel (10 lớp từ 0 đến 9).
  - **Train Set (90% của tập train gốc):** 54,000 ảnh.
  - **Validation Set (10% của tập train gốc):** 6,000 ảnh (tách cố định bằng `torch.Generator().manual_seed(seed)`).
  - **Test Set:** 10,000 ảnh (giữ nguyên gốc, chỉ đánh giá đúng 1 lần).
- **Tiền xử lý:** Chuẩn hóa pixel về đoạn $[0, 1]$ qua `ToTensor()` và chuẩn hóa phân phối $Z$-score với `mean = (0.1307,)`, `std = (0.3081,)`.

### 2.2. Bảng So Sánh Hai Bản (Cùng siêu tham số: LR=0.1, Batch=128, Epochs=30, Patience=5)
| Phiên bản | Test Accuracy (%) | Test Loss | Thời gian / Epoch (s) | Cơ chế cập nhật |
| :--- | :---: | :---: | :---: | :--- |
| **Bản 1: Tensor + Autograd** | $\ge 97.5\%$ | $\approx 0.08$ | Nhanh, trực tiếp | 1 dòng toán thủ công `p -= lr * p.grad` |
| **Bản 2: Thư viện `torch.nn`** | $\ge 97.8\%$ | $\approx 0.07$ | Tương đương | `optimizer.step()` |

### 2.3. Bảng Khảo Sát 3 Cách Khởi Tạo Trọng Số
| Phương pháp khởi tạo | Khởi tạo toán học | Hiện tượng quan sát được | Kết quả hội tụ |
| :--- | :--- | :--- | :--- |
| **Toàn số 0 (`zeros`)** | $W = 0, b = 0$ | Mất tính phá vỡ đối xứng (Symmetry Problem), tất cả nơ-ron nhận gradient y hệt nhau | **Thất bại hoàn toàn** (Val Acc kẹt ở mức $\approx 10\%-11\%$) |
| **Quá lớn (`large`)** | $W \sim \mathcal{N}(0, 5.0)$ | Logits cực lớn, làm hàm softmax bị bão hòa, loss ban đầu vọt cao | Rất khó hội tụ, dao động mạnh |
| **Mặc định (`default`)** | Kaiming / He Uniform | Phương sai đầu ra ổn định qua các tầng mạng | **Hội tụ nhanh, mượt mà** ($\ge 97.8\%$) |

---

## 3. Trả Lời Các Câu Hỏi Kiểm Tra P1

### Câu 1: `autograd` làm gì khi gọi `loss.backward()`? Gradient được lưu ở đâu?
- Khi chạy Forward pass, PyTorch ghi lại toàn bộ đồ thị có hướng không chu trình (DAG computation graph) kết nối từ các tensor đầu vào đến `loss`.
- Khi gọi `loss.backward()`, `autograd` duyệt ngược DAG từ nút `loss` về các nút lá có `requires_grad=True` và áp dụng Quy tắc đạo hàm hàm hợp (Chain Rule).
- Gradient đạo hàm riêng $\frac{\partial \mathcal{L}}{\partial W}$ được lưu trực tiếp vào thuộc tính **`.grad`** của chính tensor đó (ví dụ: `W1.grad`, `b1.grad`).

### Câu 2: Vì sao phải xóa gradient cũ (`zero_grad`)? Quên thì chuyện gì xảy ra?
- PyTorch mặc định **cộng dồn gradient** (`param.grad += new_grad`) để hỗ trợ kỹ thuật Gradient Accumulation khi batch size quá lớn.
- Nếu không gọi `zero_grad()` trước mỗi batch mới, gradient của batch hiện tại sẽ bị cộng đè lên gradient của tất cả các batch trước đó. Bước nhảy cập nhật sẽ bị phóng đại khổng lồ và sai lệch hướng $\rightarrow$ Loss vọt lên `NaN` và mô hình phân kỳ hoàn toàn.

### Câu 3: `nn.Linear` chứa những tham số nào và mặc định khởi tạo ra sao?
- `nn.Linear(in_features, out_features)` chứa 2 tham số:
  1. Trọng số ma trận `weight` có kích thước `(out_features, in_features)`.
  2. Vector `bias` có kích thước `(out_features,)`.
- Mặc định PyTorch khởi tạo cả `weight` và `bias` theo phân phối đều **Kaiming Uniform** trong khoảng $[-\frac{1}{\sqrt{\text{in\_features}}}, \frac{1}{\sqrt{\text{in\_features}}}]$.

### Câu 4: `optimizer.zero_grad()` và `optimizer.step()` tương ứng với dòng nào trong bản Tensor + Autograd?
- `optimizer.zero_grad()` $\longleftrightarrow$ `param.grad.zero_()` (duyệt qua từng tensor trọng số và xóa thuộc tính `.grad` về 0).
- `optimizer.step()` $\longleftrightarrow$ `with torch.no_grad(): param -= lr * param.grad` (công thức cập nhật độ dốc giảm dần SGD).
