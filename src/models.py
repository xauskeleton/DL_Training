import math
from typing import Any, Dict, Iterator
import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    Bản 2 (Thư viện chuẩn PyTorch):
    MLP 2 lớp ẩn xây dựng bằng nn.Module, nn.Linear, ReLU.
    Hỗ trợ 3 chế độ khởi tạo: 'default', 'zeros', 'large'.
    """

    def __init__(
        self,
        in_features: int = 784,
        hidden_1: int = 256,
        hidden_2: int = 128,
        num_classes: int = 10,
        init_mode: str = "default",
    ):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_1, hidden_2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_2, num_classes)

        self.init_mode = init_mode
        self._init_weights(init_mode)

    def _init_weights(self, mode: str) -> None:
        """3 cách khởi tạo trọng số theo yêu cầu mục 4 của P1."""
        mode = mode.lower()
        if mode == "default":
            # nn.Linear tự động khởi tạo Kaiming Uniform theo mặc định
            return

        for m in self.modules():
            if isinstance(m, nn.Linear):
                if mode == "zeros":
                    nn.init.zeros_(m.weight)
                    nn.init.zeros_(m.bias)
                elif mode == "large":
                    nn.init.normal_(m.weight, mean=0.0, std=5.0)
                    nn.init.zeros_(m.bias)
                else:
                    raise ValueError(
                        f"Chế độ khởi tạo '{mode}' không hợp lệ. Chọn: default, zeros, large"
                    )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        logits = self.fc3(x)
        return logits


class TensorMLP:
    """
    Bản 1 (Tensor + Autograd thuần túy):
    - Trọng số tự khai báo bằng tensor PyTorch với requires_grad=True.
    - Forward pass tự tính bằng phép nhân ma trận (@) và ReLU.
    - Tự tính Cross-Entropy loss bằng Log-Sum-Exp ổn định số học.
    - Cập nhật trọng số bằng 1 dòng SGD thuần: param -= lr * param.grad.
    - Xóa gradient thủ công: param.grad.zero_().
    - Tương thích giao diện với bộ khung chuẩn (state_dict, to, eval, parameters).
    """

    def __init__(
        self,
        in_features: int = 784,
        hidden_1: int = 256,
        hidden_2: int = 128,
        num_classes: int = 10,
        device: torch.device = None,
    ):
        device = device or torch.device("cpu")
        self.device = device
        self.training = True

        # Khởi tạo trọng số bằng Kaiming / He Normal thuần túy
        self.W1 = (
            torch.randn(in_features, hidden_1, device=device) * math.sqrt(2.0 / in_features)
        ).requires_grad_(True)
        self.b1 = torch.zeros(hidden_1, device=device, requires_grad=True)

        self.W2 = (
            torch.randn(hidden_1, hidden_2, device=device) * math.sqrt(2.0 / hidden_1)
        ).requires_grad_(True)
        self.b2 = torch.zeros(hidden_2, device=device, requires_grad=True)

        self.W3 = (
            torch.randn(hidden_2, num_classes, device=device) * math.sqrt(2.0 / hidden_2)
        ).requires_grad_(True)
        self.b3 = torch.zeros(num_classes, device=device, requires_grad=True)

        self.params = [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        X = x.view(x.size(0), -1).to(self.device)
        h1 = torch.relu(X @ self.W1 + self.b1)
        h2 = torch.relu(h1 @ self.W2 + self.b2)
        logits = h2 @ self.W3 + self.b3
        return logits

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

    def compute_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Hàm Cross-Entropy Loss viết tay bằng Tensor thuần (Log-Sum-Exp trick)."""
        targets = targets.to(self.device)
        max_logits = logits.max(dim=1, keepdim=True).values
        stable_logits = logits - max_logits
        sum_exp = torch.exp(stable_logits).sum(dim=1, keepdim=True)
        log_probs = stable_logits - torch.log(sum_exp)
        nll = -log_probs[torch.arange(targets.size(0), device=self.device), targets]
        return nll.mean()

    def step_sgd(self, lr: float) -> None:
        """Cập nhật trọng số bằng 1 dòng SGD thuần túy & xóa gradient thủ công."""
        with torch.no_grad():
            for p in self.params:
                if p.grad is not None:
                    p -= lr * p.grad
                    p.grad.zero_()

    def parameters(self) -> Iterator[torch.Tensor]:
        return iter(self.params)

    def train(self, mode: bool = True) -> "TensorMLP":
        self.training = mode
        return self

    def eval(self) -> "TensorMLP":
        self.training = False
        return self

    def to(self, device: torch.device) -> "TensorMLP":
        self.device = device
        with torch.no_grad():
            self.W1 = self.W1.to(device).requires_grad_(True)
            self.b1 = self.b1.to(device).requires_grad_(True)
            self.W2 = self.W2.to(device).requires_grad_(True)
            self.b2 = self.b2.to(device).requires_grad_(True)
            self.W3 = self.W3.to(device).requires_grad_(True)
            self.b3 = self.b3.to(device).requires_grad_(True)
            self.params = [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]
        return self

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {
            "W1": self.W1.clone().detach(),
            "b1": self.b1.clone().detach(),
            "W2": self.W2.clone().detach(),
            "b2": self.b2.clone().detach(),
            "W3": self.W3.clone().detach(),
            "b3": self.b3.clone().detach(),
        }

    def load_state_dict(self, state: Dict[str, torch.Tensor]) -> None:
        with torch.no_grad():
            self.W1.copy_(state["W1"])
            self.b1.copy_(state["b1"])
            self.W2.copy_(state["W2"])
            self.b2.copy_(state["b2"])
            self.W3.copy_(state["W3"])
            self.b3.copy_(state["b3"])


def build_model(config: Dict[str, Any]) -> Any:
    """
    Khởi tạo mô hình dựa vào cấu hình:
    - model.name = 'tensor_mlp': Trả về Bản 1 (Tensor + Autograd thuần túy).
    - model.name = 'mlp': Trả về Bản 2 (Thư viện chuẩn nn.Module).
    """
    model_cfg = config.get("model", {})
    name = model_cfg.get("name", "mlp").lower()
    hidden_1 = model_cfg.get("hidden_1", 256)
    hidden_2 = model_cfg.get("hidden_2", 128)
    num_classes = config.get("data", {}).get("num_classes", 10)
    init_mode = model_cfg.get("init_mode", "default")

    if name in ("tensor_mlp", "tensor_autograd"):
        return TensorMLP(
            in_features=784,
            hidden_1=hidden_1,
            hidden_2=hidden_2,
            num_classes=num_classes,
        )

    # Mặc định dùng bản nn.Module
    return MLP(
        in_features=784,
        hidden_1=hidden_1,
        hidden_2=hidden_2,
        num_classes=num_classes,
        init_mode=init_mode,
    )
