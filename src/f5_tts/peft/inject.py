from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from torch import nn

from f5_tts.peft.conditioning import ConditioningConvAdapter
from f5_tts.peft.lora import LoRALinear


@dataclass
class PVCAdapterConfig:
    rank: int = 16
    alpha: float = 16.0
    lora_dropout: float = 0.05
    prompt_drop_path: float = 0.3
    prompt_target: str = "transformer.input_embed.proj"
    dit_target_regex: str = r"^transformer\.transformer_blocks\.\d+\.attn\.to_(q|v)$"
    conditioning_enabled: bool = False
    conditioning_gamma: float = 0.25
    conditioning_kernel_size: int = 3
    conditioning_se_reduction: int = 4
    conditioning_target_regex: str = r"^transformer\.text_embed\.text_blocks\.\d+\.dwconv$"

    def to_dict(self):
        return asdict(self)


def freeze_model_parameters(model: nn.Module):
    for p in model.parameters():
        p.requires_grad = False


def _resolve_parent_and_child(root: nn.Module, module_name: str):
    parts = module_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def _replace_linear_with_lora(
    model: nn.Module,
    module_name: str,
    rank: int,
    alpha: float,
    lora_dropout: float,
    branch_drop: float,
):
    parent, child_name = _resolve_parent_and_child(model, module_name)
    layer = getattr(parent, child_name)
    if isinstance(layer, LoRALinear):
        return True
    if not isinstance(layer, nn.Linear):
        raise TypeError(f"Target module is not nn.Linear: {module_name} ({type(layer)})")

    wrapped = LoRALinear(
        layer,
        rank=rank,
        alpha=alpha,
        lora_dropout=lora_dropout,
        branch_drop=branch_drop,
    )
    setattr(parent, child_name, wrapped)
    return True


def _replace_conv1d_with_conditioning_adapter(
    model: nn.Module,
    module_name: str,
    gamma: float,
    kernel_size: int,
    se_reduction: int,
):
    parent, child_name = _resolve_parent_and_child(model, module_name)
    layer = getattr(parent, child_name)
    if isinstance(layer, ConditioningConvAdapter):
        return True
    if not isinstance(layer, nn.Conv1d):
        raise TypeError(f"Target module is not nn.Conv1d: {module_name} ({type(layer)})")

    wrapped = ConditioningConvAdapter(
        layer,
        gamma=gamma,
        kernel_size=kernel_size,
        se_reduction=se_reduction,
    )
    setattr(parent, child_name, wrapped)
    return True


def apply_pvc_adapters(model: nn.Module, cfg: PVCAdapterConfig):
    prompt_modules = []
    dit_modules = []
    conditioning_modules = []

    module_names = [name for name, _ in model.named_modules()]
    for name in module_names:
        if name == cfg.prompt_target:
            replaced = _replace_linear_with_lora(
                model,
                name,
                rank=cfg.rank,
                alpha=cfg.alpha,
                lora_dropout=cfg.lora_dropout,
                branch_drop=cfg.prompt_drop_path,
            )
            if replaced:
                prompt_modules.append(name)
        elif re.match(cfg.dit_target_regex, name):
            replaced = _replace_linear_with_lora(
                model,
                name,
                rank=cfg.rank,
                alpha=cfg.alpha,
                lora_dropout=cfg.lora_dropout,
                branch_drop=0.0,
            )
            if replaced:
                dit_modules.append(name)
        elif cfg.conditioning_enabled and re.match(cfg.conditioning_target_regex, name):
            replaced = _replace_conv1d_with_conditioning_adapter(
                model,
                name,
                gamma=cfg.conditioning_gamma,
                kernel_size=cfg.conditioning_kernel_size,
                se_reduction=cfg.conditioning_se_reduction,
            )
            if replaced:
                conditioning_modules.append(name)

    if not prompt_modules:
        raise RuntimeError(f"Prompt target not found or not replaced: {cfg.prompt_target}")
    if not dit_modules:
        raise RuntimeError(f"No DiT target modules replaced with regex: {cfg.dit_target_regex}")
    if cfg.conditioning_enabled and not conditioning_modules:
        raise RuntimeError(f"No conditioning modules replaced with regex: {cfg.conditioning_target_regex}")

    return {
        "prompt_modules": prompt_modules,
        "dit_modules": dit_modules,
        "conditioning_modules": conditioning_modules,
        "all_modules": prompt_modules + dit_modules + conditioning_modules,
    }


def count_parameters(model: nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def set_lora_strength(model: nn.Module, strength: float):
    strength = float(strength)
    updated = 0
    for _, module in model.named_modules():
        if isinstance(module, (LoRALinear, ConditioningConvAdapter)):
            module.runtime_scale = strength
            updated += 1
    return updated
