from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class SqueezeExcite1d(nn.Module):
    """Squeeze-and-Excitation gate for 1D feature maps."""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        if channels <= 0:
            raise ValueError(f"channels must be > 0, got {channels}")
        if reduction <= 0:
            raise ValueError(f"reduction must be > 0, got {reduction}")

        hidden = max(1, channels // reduction)
        self.reduce = nn.Conv1d(channels, hidden, kernel_size=1, bias=True)
        self.expand = nn.Conv1d(hidden, channels, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Global average pooling over time dimension.
        s = x.mean(dim=-1, keepdim=True)
        s = self.reduce(s)
        s = F.silu(s)
        s = self.expand(s)
        s = torch.sigmoid(s)
        return x * s


class ConditioningConvAdapter(nn.Module):
    """
    Conv-Adapter attached to ConvNeXtV2 depth-wise conv layers.

    Adapter path:
      point-wise projection (compression by gamma)
      -> depth-wise conv (kernel_size)
      -> point-wise projection (restore channels)
      -> SE gating
    """

    def __init__(
        self,
        base_layer: nn.Conv1d,
        gamma: float = 0.25,
        kernel_size: int = 3,
        se_reduction: int = 4,
    ):
        super().__init__()
        if not isinstance(base_layer, nn.Conv1d):
            raise TypeError(f"base_layer must be nn.Conv1d, got {type(base_layer)}")
        if base_layer.in_channels != base_layer.out_channels:
            raise ValueError(
                "ConditioningConvAdapter requires in_channels == out_channels, "
                f"got {base_layer.in_channels} and {base_layer.out_channels}"
            )
        if base_layer.groups != base_layer.in_channels:
            raise ValueError(
                "ConditioningConvAdapter targets depth-wise conv only, "
                f"but groups={base_layer.groups}, in_channels={base_layer.in_channels}"
            )
        if gamma <= 0:
            raise ValueError(f"gamma must be > 0, got {gamma}")
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be a positive odd number, got {kernel_size}")

        self.base_layer = base_layer
        self.gamma = float(gamma)
        self.kernel_size = int(kernel_size)
        self.se_reduction = int(se_reduction)
        self.runtime_scale = 1.0

        channels = base_layer.in_channels
        hidden = max(1, int(round(channels * self.gamma)))
        param_device = base_layer.weight.device
        param_dtype = base_layer.weight.dtype

        self.pw_down = nn.Conv1d(channels, hidden, kernel_size=1, bias=False)
        self.dw = nn.Conv1d(
            hidden,
            hidden,
            kernel_size=self.kernel_size,
            padding=self.kernel_size // 2,
            groups=hidden,
            bias=False,
        )
        self.pw_up = nn.Conv1d(hidden, channels, kernel_size=1, bias=False)
        self.se = SqueezeExcite1d(channels=channels, reduction=self.se_reduction)
        self.act = nn.GELU()

        self.to(device=param_device, dtype=param_dtype)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.pw_down.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.dw.weight, a=math.sqrt(5))
        # Zero-init to preserve pre-trained behavior at step 0.
        nn.init.zeros_(self.pw_up.weight)

        nn.init.kaiming_uniform_(self.se.reduce.weight, a=math.sqrt(5))
        nn.init.zeros_(self.se.reduce.bias)
        nn.init.zeros_(self.se.expand.weight)
        nn.init.zeros_(self.se.expand.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base_layer(x)
        a = self.pw_down(x)
        a = self.dw(a)
        a = self.act(a)
        a = self.pw_up(a)
        a = self.se(a)
        return out + a * self.runtime_scale

    def export_adapter_state(self):
        return {
            "pw_down.weight": self.pw_down.weight.detach().cpu().contiguous(),
            "dw.weight": self.dw.weight.detach().cpu().contiguous(),
            "pw_up.weight": self.pw_up.weight.detach().cpu().contiguous(),
            "se.reduce.weight": self.se.reduce.weight.detach().cpu().contiguous(),
            "se.reduce.bias": self.se.reduce.bias.detach().cpu().contiguous(),
            "se.expand.weight": self.se.expand.weight.detach().cpu().contiguous(),
            "se.expand.bias": self.se.expand.bias.detach().cpu().contiguous(),
        }

    def load_adapter_state(self, state: dict, prefix: str):
        mapping = {
            "pw_down.weight": self.pw_down.weight,
            "dw.weight": self.dw.weight,
            "pw_up.weight": self.pw_up.weight,
            "se.reduce.weight": self.se.reduce.weight,
            "se.reduce.bias": self.se.reduce.bias,
            "se.expand.weight": self.se.expand.weight,
            "se.expand.bias": self.se.expand.bias,
        }
        missing = []
        consumed = set()
        for key, target in mapping.items():
            full_key = f"{prefix}.cond.{key}"
            value = state.get(full_key, None)
            if value is None:
                missing.append(full_key)
                continue
            target.data.copy_(value.to(device=target.device, dtype=target.dtype))
            consumed.add(full_key)
        return missing, consumed

    def clear_adapter(self):
        self.pw_down.weight.data.zero_()
        self.dw.weight.data.zero_()
        self.pw_up.weight.data.zero_()
        self.se.reduce.weight.data.zero_()
        self.se.reduce.bias.data.zero_()
        self.se.expand.weight.data.zero_()
        self.se.expand.bias.data.zero_()
