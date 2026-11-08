import argparse
import time
import torch
import torch.nn.functional as F

from src.models import build_model
from src.utils import get_device, load_checkpoint, load_config


def parse_args():
    parser = argparse.ArgumentParser(description="Chạy suy luận (Inference) trên mẫu ngoài bộ dữ liệu")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="File cấu hình")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="outputs/checkpoints/best_model_seed_42.pt",
        help="Đường dẫn tới file checkpoint tốt nhất",
    )
    parser.add_argument("--input", type=str, required=True, help="Đường dẫn ảnh/văn bản đầu vào")
    return parser.parse_args()


def preprocess_input(input_path: str) -> torch.Tensor:
    """Tiền xử lý đầu vào mới (ảnh tự chụp / văn bản mới) về đúng tensor chuẩn của model."""
    raise NotImplementedError("Cài đặt hàm preprocess_input() cho từng project.")


def main():
    args = parse_args()
    config = load_config(args.config)
    device = get_device()

    model = build_model(config).to(device)
    load_checkpoint(model, args.checkpoint, device)
    model.eval()

    x = preprocess_input(args.input).to(device)

    # Đo thời gian suy luận (ms)
    start_time = time.perf_counter()
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=-1)
    latency_ms = (time.perf_counter() - start_time) * 1000.0

    pred_class = probs.argmax(dim=-1).item()
    confidence = probs.max(dim=-1).values.item() * 100.0

    print(f"Input: {args.input}")
    print(f"Dự đoán: Lớp {pred_class} (Độ tin cậy: {confidence:.2f}%)")
    print(f"Thời gian suy luận: {latency_ms:.2f} ms")


if __name__ == "__main__":
    main()

