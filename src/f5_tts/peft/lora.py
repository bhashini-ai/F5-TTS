from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from f5_tts.peft.drop_path import DropPath


class LoRALinear(nn.Module):
    """LoRA wrapper for nn.Linear with optional branch DropPath."""

    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int = 16,
        alpha: float = 16.0,
        lora_dropout: float = 0.0,
        branch_drop: float = 0.0,
    ):
        super().__init__()
        if rank <= 0:
            raise ValueError(f"rank must be > 0, got {rank}")
        if not isinstance(base_layer, nn.Linear):
            raise TypeError(f"base_layer must be nn.Linear, got {type(base_layer)}")

        self.base_layer = base_layer
        self.rank = rank
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.merged = False

        self.lora_dropout = nn.Dropout(lora_dropout) if lora_dropout > 0 else nn.Identity()
        self.branch_drop = DropPath(branch_drop) if branch_drop > 0 else nn.Identity()

        self.lora_A = nn.Parameter(torch.empty(rank, base_layer.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base_layer.out_features, rank))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base_layer(x)
        if self.merged:
            return out

        lora = F.linear(self.lora_dropout(x), self.lora_A)
        lora = F.linear(lora, self.lora_B)
        lora = self.branch_drop(lora * self.scaling)
        return out + lora

    @torch.no_grad()
    def merge(self):
        if self.merged:
            return
        delta_w = (self.lora_B @ self.lora_A) * self.scaling
        self.base_layer.weight.data.add_(delta_w.to(self.base_layer.weight.dtype))
        self.merged = True

    @torch.no_grad()
    def unmerge(self):
        if not self.merged:
            return
        delta_w = (self.lora_B @ self.lora_A) * self.scaling
        self.base_layer.weight.data.sub_(delta_w.to(self.base_layer.weight.dtype))
        self.merged = False
