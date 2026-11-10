import argparse
import os
from pathlib import Path
import time
from typing import Tuple
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps
import torch
import torch.nn.functional as F
from torchvision import transforms

from src.models import build_model
from src.utils import get_device, load_checkpoint, load_config


def parse_args():
    parser = argparse.ArgumentParser(description="Chạy suy luận (Inference) trên ảnh chữ số viết tay")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="File cấu hình")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="outputs/checkpoints/best_model_seed_42.pt",
        help="Đường dẫn checkpoint",
    )
    parser.add_argument("--input", type=str, required=True, help="Đường dẫn ảnh chữ số đầu vào")
    parser.add_argument(
        "--save_result",
        type=str,
        default="outputs/figures/inference_result.png",
        help="Đường dẫn lưu ảnh kết quả trực quan hóa",
    )
    return parser.parse_args()


def preprocess_input(input_path: str) -> Tuple[torch.Tensor, Image.Image]:
    """
    Tiền xử lý ảnh chữ số viết tay bên ngoài:
    1. Chuyển sang ảnh thang xám (Grayscale).
    2. Tự động phát hiện và đảo màu nếu ảnh là 'chữ đen nền trắng'
       thành 'chữ trắng nền đen' cho chuẩn với định dạng MNIST.
    3. Đưa kích thước về 28x28 pixel.
    4. Chuẩn hóa bằng mean=0.1307, std=0.3081.
    """
    img = Image.open(input_path).convert("L")

    # Kiểm tra màu nền: Nếu trung bình pixel > 127 (nền sáng chữ tối), đảo màu lại
    arr = np.array(img)
    if arr.mean() > 127:
        img = ImageOps.invert(img)

    img_resized = img.resize((28, 28), Image.Resampling.LANCZOS)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    tensor = transform(img_resized).unsqueeze(0)  # Thêm chiều batch: [1, 1, 28, 28]
    return tensor, img_resized


def main():
    args = parse_args()
    config = load_config(args.config)
    device = get_device()

    model = build_model(config).to(device)
    load_checkpoint(model, args.checkpoint, device)
    model.eval()

    tensor, preview_img = preprocess_input(args.input)
    tensor = tensor.to(device)

    # Đo thời gian suy luận (ms)
    start_time = time.perf_counter()
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=-1)
    latency_ms = (time.perf_counter() - start_time) * 1000.0

    probs_np = probs.squeeze(0).cpu().numpy()
    pred_class = int(probs_np.argmax())
    confidence = float(probs_np[pred_class]) * 100.0

    print("\n================ KẾT QUẢ SUY LUẬN (INFERENCE) ================")
    print(f"Ảnh đầu vào: {args.input}")
    print(f"Dự đoán: CHỮ SỐ {pred_class} (Độ tin cậy: {confidence:.2f}%)")
    print(f"Thời gian suy luận (Latency): {latency_ms:.2f} ms")
    print("Xác suất phân phối Top 3:")
    top3_indices = probs_np.argsort()[-3:][::-1]
    for rank, idx in enumerate(top3_indices, 1):
        print(f"  {rank}. Chữ số {idx}: {probs_np[idx] * 100:.2f}%")
    print("==============================================================")

    # Lưu hình ảnh trực quan hóa
    if args.save_result:
        Path(os.path.dirname(args.save_result)).mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))

        axes[0].imshow(preview_img, cmap="gray")
        axes[0].set_title(f"Ảnh sau tiền xử lý (28x28)")
        axes[0].axis("off")

        axes[1].bar(range(10), probs_np * 100, color="steelblue")
        axes[1].set_title(f"Dự đoán: Số {pred_class} ({confidence:.1f}%)")
        axes[1].set_xlabel("Chữ số (0-9)")
        axes[1].set_ylabel("Xác suất (%)")
        axes[1].set_xticks(range(10))
        axes[1].set_ylim(0, 100)

        plt.tight_layout()
        plt.savefig(args.save_result, dpi=150)
        plt.close(fig)
        print(f"Đã lưu hình ảnh trực quan suy luận tại: {args.save_result}")


if __name__ == "__main__":
    main()
