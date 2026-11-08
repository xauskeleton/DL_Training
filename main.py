
import argparse
import os
import torch.nn as nn
import torch.optim as optim

from src.dataset import get_dataloaders
from src.engine import evaluate, fit
from src.models import build_model
from src.sanity_check import check_initial_loss, check_overfit_single_batch
from src.utils import (
    build_optimizer,
    build_scheduler,
    count_parameters,
    get_device,
    load_config,
    set_seed,
    summarize_seeds,
)
from src.visualize import plot_confusion_matrix, plot_learning_curves


def parse_args():
    parser = argparse.ArgumentParser(description="Khung huấn luyện Deep Learning chuẩn")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Đường dẫn tới file cấu hình YAML",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    device = get_device()
    print(f"=== Project: {config['project_name']} | Device: {device} ===")

    seeds = config["train"]["seeds"]
    test_accuracies = []

    for idx, seed in enumerate(seeds):
        print(f"\n--- Run {idx + 1}/{len(seeds)} với Seed = {seed} ---")
        set_seed(seed)

        # 1. Nạp dữ liệu
        train_loader, val_loader, test_loader = get_dataloaders(
            data_dir=config["data"]["data_dir"],
            batch_size=config["data"]["batch_size"],
            val_ratio=config["data"]["val_ratio"],
            seed=seed,
        )

        # 2. Khởi tạo mô hình, hàm mất mát và bộ tối ưu
        model = build_model(config).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = build_optimizer(model, config)
        scheduler = build_scheduler(
            optimizer=optimizer,
            config=config,
            epochs=config["train"]["epochs"],
        )

        # 3. Chạy Sanity Check ở seed đầu tiên trước khi huấn luyện thật
        if idx == 0:
            print(f"Số tham số huấn luyện: {count_parameters(model):,}")
            check_initial_loss(
                model, train_loader, criterion, config["data"]["num_classes"], device
            )
            check_overfit_single_batch(model, train_loader, criterion, device)

        # 4. Huấn luyện kèm Early Stopping trên tập Validation
        ckpt_path = os.path.join(
            config["output"]["checkpoint_dir"], f"best_model_seed_{seed}.pt"
        )
        log_dir = os.path.join(config["output"]["log_dir"], f"seed_{seed}")

        history = fit(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epochs=config["train"]["epochs"],
            patience=config["train"]["patience"],
            checkpoint_path=ckpt_path,
            log_dir=log_dir,
            scheduler=scheduler,
        )

        # 5. Đánh giá đúng 1 lần trên tập Test bằng checkpoint tốt nhất
        test_loss, test_acc, y_pred, y_true = evaluate(
            model, test_loader, criterion, device
        )
        print(f"[Seed {seed}] Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.2f}%")
        test_accuracies.append(test_acc)

        # Lưu biểu đồ ở seed đầu tiên
        if idx == 0:
            plot_learning_curves(
                history,
                save_path=os.path.join(config["output"]["figure_dir"], "learning_curves.png"),
            )
            plot_confusion_matrix(
                y_true,
                y_pred,
                save_path=os.path.join(config["output"]["figure_dir"], "confusion_matrix.png"),
            )

    # 6. Tổng hợp kết quả qua các seed (mean ± std)
    print("\n==========================================")
    print(f"Kết quả Test Accuracy ({len(seeds)} seeds): {summarize_seeds(test_accuracies)}%")
    print("==========================================")


if __name__ == "__main__":
    main()

