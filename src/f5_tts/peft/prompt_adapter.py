from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class PromptAdapterConfig:
    target_module: str = "transformer.input_embed.proj"
    rank: int = 16
    alpha: float = 16.0
    lora_dropout: float = 0.05
    drop_path: float = 0.3

    def to_dict(self):
        return asdict(self)
