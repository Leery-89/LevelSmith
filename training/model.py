"""
LevelSmith 结构参数预测 MLP 模型

架构: 16 → 128 → 64 → 32 → 23
- BatchNorm + Dropout 防止过拟合
- Sigmoid 输出层确保输出在 [0,1] (归一化参数空间)
- 输出维度: 原有10 + 视觉10 + 几何复杂度3 (mesh_complexity/detail_density/simple_ratio)
"""

import torch
import torch.nn as nn
from typing import List, Optional


class StyleParamMLP(nn.Module):
    """
    风格特征向量 → 结构参数预测网络

    Args:
        input_dim:   特征向量维度 (默认 16)
        output_dim:  输出参数数量 (默认 23)
        hidden_dims: 隐藏层维度列表 (默认 [128, 64, 32])
        dropout:     Dropout 概率
    """

    def __init__(
        self,
        input_dim: int = 16,
        output_dim: int = 23,
        hidden_dims: List[int] = None,
        dropout: float = 0.2,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64, 32]

        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers += [
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ]
            in_dim = h_dim

        layers.append(nn.Linear(in_dim, output_dim))
        layers.append(nn.Sigmoid())  # 输出归一化到 [0,1]

        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# subdivision 在 OUTPUT_KEYS 中的固定位置（原有第10个参数，索引=9）
_SUBDIV_IDX = 9

class StyleParamLoss(nn.Module):
    """
    自定义损失函数：MSE + subdivision 整数惩罚项

    subdivision 位于输出向量第 _SUBDIV_IDX 维（固定为索引9，新增参数追加在其后）。
    """

    def __init__(self, subdiv_weight: float = 0.1):
        super().__init__()
        self.mse = nn.MSELoss()
        self.subdiv_weight = subdiv_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mse_loss = self.mse(pred, target)

        # subdivision 整数对齐惩罚（归一化空间中步长 ≈ 1/7，范围1-8共7步）
        subdiv_pred = pred[:, _SUBDIV_IDX]
        step = 1.0 / 7.0
        subdiv_rounded = torch.round(subdiv_pred / step) * step
        subdiv_penalty = self.mse(subdiv_pred, subdiv_rounded.detach())

        return mse_loss + self.subdiv_weight * subdiv_penalty


def build_model(
    input_dim: int = 16,
    output_dim: int = 20,
    hidden_dims: List[int] = None,
    dropout: float = 0.2,
    device: str = "cuda",
) -> StyleParamMLP:
    """创建模型并移动到指定设备"""
    if hidden_dims is None:
        hidden_dims = [128, 64, 32]
    model = StyleParamMLP(input_dim, output_dim, hidden_dims, dropout)
    model = model.to(device)
    return model


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def clamp_output_to_style_bounds(
    pred: torch.Tensor,
    style_name: str,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    推理时将模型输出 clamp 到指定风格的归一化边界内。

    Sigmoid 输出已保证全局 [0,1]，本函数在此基础上施加更紧的风格级约束，
    防止预测值越界到其他风格的参数空间。

    Args:
        pred:       模型输出，shape [..., OUTPUT_DIM]，值域 [0,1]
        style_name: 目标风格名称，需在 STYLE_REGISTRY 中存在
        device:     目标设备，默认与 pred 相同

    Returns:
        clamp 后的张量，shape 与 pred 相同
    """
    import numpy as np
    # 延迟导入避免循环依赖
    from style_registry import get_style_bounds_normalized

    bounds_np = get_style_bounds_normalized(style_name)  # [OUTPUT_DIM, 2]
    target_device = device if device is not None else pred.device
    lo = torch.from_numpy(bounds_np[:, 0]).to(target_device)
    hi = torch.from_numpy(bounds_np[:, 1]).to(target_device)
    return torch.clamp(pred, min=lo, max=hi)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model = build_model(device=device)
    print(f"\n模型结构:\n{model}")
    print(f"\n可训练参数: {count_parameters(model):,}")

    # 前向传播测试
    dummy = torch.randn(4, 16).to(device)
    out = model(dummy)
    print(f"\n输入形状: {dummy.shape} → 输出形状: {out.shape}")
    print(f"输出值域: [{out.min().item():.4f}, {out.max().item():.4f}]")
