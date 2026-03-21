from __future__ import annotations

import json
import os
import hashlib

from safetensors.torch import load_file, save_file

from f5_tts.peft.conditioning import ConditioningConvAdapter
from f5_tts.peft.lora import LoRALinear


def extract_adapter_state_dict(model):
    state = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            state[f"{name}.lora_A"] = module.lora_A.detach().cpu().contiguous()
            state[f"{name}.lora_B"] = module.lora_B.detach().cpu().contiguous()
        elif isinstance(module, ConditioningConvAdapter):
            for key, value in module.export_adapter_state().items():
                state[f"{name}.cond.{key}"] = value
    if not state:
        raise RuntimeError("No adapter modules found. Did you inject adapters before saving?")
    return state


def save_adapter(model, output_dir: str, adapter_config: dict):
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "adapter_model.safetensors")
    cfg_path = os.path.join(output_dir, "adapter_config.json")

    state = extract_adapter_state_dict(model)
    save_file(state, model_path)

    payload = {"format_version": 1, **adapter_config}
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def load_adapter(model, adapter_dir: str, strict: bool = True):
    state, config = load_adapter_state(adapter_dir)
    return apply_adapter_state(model, state, config=config, strict=strict)


def load_adapter_state(adapter_dir: str, device: str = "cpu"):
    model_path = os.path.join(adapter_dir, "adapter_model.safetensors")
    cfg_path = os.path.join(adapter_dir, "adapter_config.json")
    state = load_file(model_path, device=device)
    config = {}
    if os.path.exists(cfg_path):
        with open(cfg_path, "r", encoding="utf-8") as f:
            config = json.load(f)
    return state, config


def apply_adapter_state(model, state, config: dict | None = None, strict: bool = True):
    missing = []
    consumed = set()
    has_conditioning_state = any(".cond." in key for key in state.keys())
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            key_a = f"{name}.lora_A"
            key_b = f"{name}.lora_B"
            if key_a not in state or key_b not in state:
                missing.append(name)
                continue
            module.lora_A.data.copy_(state[key_a].to(device=module.lora_A.device, dtype=module.lora_A.dtype))
            module.lora_B.data.copy_(state[key_b].to(device=module.lora_B.device, dtype=module.lora_B.dtype))
            consumed.add(key_a)
            consumed.add(key_b)
        elif isinstance(module, ConditioningConvAdapter):
            # Backward compatibility: old adapters may not include conditioning weights.
            if not has_conditioning_state:
                module.clear_adapter()
                continue
            cond_missing, cond_consumed = module.load_adapter_state(state, prefix=name)
            if cond_missing:
                missing.append(name)
                continue
            consumed.update(cond_consumed)

    unexpected = sorted(k for k in state.keys() if k not in consumed)
    if strict and (missing or unexpected):
        raise RuntimeError(
            f"Adapter load mismatch. Missing modules: {missing if missing else '[]'}, unexpected keys: {unexpected}"
        )
    return {"config": config or {}, "missing_modules": missing, "unexpected_keys": unexpected}


def clear_adapter(model):
    found = 0
    for _, module in model.named_modules():
        if isinstance(module, LoRALinear):
            module.lora_A.data.zero_()
            module.lora_B.data.zero_()
            found += 1
        elif isinstance(module, ConditioningConvAdapter):
            module.clear_adapter()
            found += 1
    return found


def file_sha256(path: str):
    if not path or not os.path.isfile(path):
        return ""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def check_adapter_compatibility(
    adapter_config: dict,
    expected_model_name: str = "",
    expected_base_ckpt_sha256: str = "",
    strict: bool = True,
):
    mismatches = []

    adapter_model_name = (
        adapter_config.get("exp_name", "")
        or adapter_config.get("model_name", "")
        or adapter_config.get("base_model_name", "")
    )
    if expected_model_name:
        if not adapter_model_name:
            mismatches.append("missing adapter model metadata (exp_name/model_name/base_model_name)")
        elif adapter_model_name != expected_model_name:
            mismatches.append(
                f"model mismatch: adapter expects '{adapter_model_name}', runtime model is '{expected_model_name}'"
            )

    adapter_sha = adapter_config.get("base_ckpt_sha256", "")
    if expected_base_ckpt_sha256:
        if not adapter_sha:
            mismatches.append("missing adapter base_ckpt_sha256 metadata")
        elif adapter_sha != expected_base_ckpt_sha256:
            mismatches.append(
                f"base ckpt hash mismatch: adapter '{adapter_sha[:12]}...', runtime '{expected_base_ckpt_sha256[:12]}...'"
            )

    if strict and mismatches:
        raise RuntimeError("Incompatible adapter: " + "; ".join(mismatches))

    return {"ok": len(mismatches) == 0, "mismatches": mismatches}
