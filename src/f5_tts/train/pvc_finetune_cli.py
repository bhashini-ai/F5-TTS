from __future__ import annotations

import argparse
import hashlib
import json
import os
from importlib.resources import files
from pathlib import Path

import torch
from cached_path import cached_path
from hydra.utils import get_class
from omegaconf import OmegaConf
from safetensors.torch import save_file

from f5_tts.infer.utils_infer import load_checkpoint
from f5_tts.model import CFM, Trainer
from f5_tts.model.dataset import load_dataset
from f5_tts.model.utils import get_tokenizer
from f5_tts.peft.inject import PVCAdapterConfig, apply_pvc_adapters, count_parameters, freeze_model_parameters
from f5_tts.peft.io import save_adapter


def parse_args():
    parser = argparse.ArgumentParser(description="PVC fine-tuning with LoRA + Prompt Adapter")

    parser.add_argument(
        "--exp_name",
        type=str,
        default="F5TTS_v1_Base",
        choices=["F5TTS_v1_Base", "F5TTS_Base", "E2TTS_Base"],
    )
    parser.add_argument("--dataset_path", type=str, required=True, help="Prepared dataset folder with raw.arrow + vocab.txt")
    parser.add_argument(
        "--vocab_file",
        type=str,
        default="",
        help="Optional vocab.txt override. Use the base-model vocab file for language-specific checkpoints.",
    )
    parser.add_argument("--speaker_id", type=str, required=True)
    parser.add_argument("--base_ckpt", type=str, default="", help="Local or hf:// checkpoint path")
    parser.add_argument("--use_ema", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--batch_size_per_gpu", type=int, default=3200)
    parser.add_argument("--batch_size_type", type=str, default="frame", choices=["frame", "sample"])
    parser.add_argument("--max_samples", type=int, default=64)
    parser.add_argument("--grad_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--num_warmup_updates", type=int, default=20000)
    parser.add_argument("--save_per_updates", type=int, default=5000)
    parser.add_argument("--keep_last_n_checkpoints", type=int, default=2)
    parser.add_argument("--last_per_updates", type=int, default=1000)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--logger", type=str, default="none", choices=["none", "wandb", "tensorboard"])
    parser.add_argument("--checkpoint_dir", type=str, default="")
    parser.add_argument(
        "--distill_weight",
        type=float,
        default=0.0,
        help="Weight for multilingual retention distillation loss (0 disables distillation).",
    )
    parser.add_argument(
        "--distill_prob",
        type=float,
        default=0.0,
        help="Probability of applying distillation on a training update.",
    )
    parser.add_argument(
        "--distill_anchor_text_file",
        type=str,
        default="",
        help="UTF-8 text file with multilingual anchor texts (one line per anchor).",
    )
    parser.add_argument(
        "--distill_student_lora_strength",
        type=float,
        default=1.0,
        help="LoRA strength used for the student pass during distillation.",
    )

    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=float, default=16.0)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--prompt_drop_path", type=float, default=0.3)

    parser.add_argument("--output_root", type=str, default="lora")

    return parser.parse_args()


def resolve_default_ckpt(exp_name: str):
    if exp_name == "F5TTS_v1_Base":
        return "hf://SWivid/F5-TTS/F5TTS_v1_Base/model_1250000.safetensors"
    if exp_name == "F5TTS_Base":
        return "hf://SWivid/F5-TTS/F5TTS_Base/model_1200000.safetensors"
    if exp_name == "E2TTS_Base":
        return "hf://SWivid/E2-TTS/E2TTS_Base/model_1200000.safetensors"
    raise ValueError(f"Unsupported exp_name: {exp_name}")


def resolve_ckpt_path(base_ckpt: str, exp_name: str):
    ckpt = base_ckpt if base_ckpt else resolve_default_ckpt(exp_name)
    if ckpt.startswith("hf://"):
        return str(cached_path(ckpt))
    return ckpt


def file_sha256(path: str):
    if not os.path.isfile(path):
        return ""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def build_adapter_metadata(
    args,
    ckpt_path: str,
    vocab_file: str,
    dataset_path: Path,
    peft_cfg: PVCAdapterConfig,
    injected,
    total_params: int,
    trainable_params: int,
    mel_spec_kwargs,
    checkpoint_update: int | None = None,
    checkpoint_file: str = "",
):
    payload = {
        "speaker_id": args.speaker_id,
        "exp_name": args.exp_name,
        "base_ckpt": ckpt_path,
        "base_ckpt_sha256": file_sha256(ckpt_path),
        "vocab_file": str(vocab_file),
        "vocab_sha256": file_sha256(str(vocab_file)),
        "use_ema": args.use_ema,
        "dataset_path": str(dataset_path),
        "peft": peft_cfg.to_dict(),
        "injected_modules": injected,
        "trainable_params": trainable_params,
        "total_params": total_params,
        "sample_rate": int(mel_spec_kwargs.target_sample_rate),
        "distill": {
            "weight": args.distill_weight,
            "prob": args.distill_prob,
            "anchor_text_file": args.distill_anchor_text_file,
            "student_lora_strength": args.distill_student_lora_strength,
        },
    }
    if checkpoint_update is not None:
        payload["checkpoint_update"] = int(checkpoint_update)
    if checkpoint_file:
        payload["checkpoint_file"] = checkpoint_file
    return payload


def save_adapter_from_ema_checkpoint(checkpoint_file: str, output_dir: Path, adapter_config: dict):
    if not checkpoint_file or not os.path.isfile(checkpoint_file):
        return False

    checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=True)
    ema_state = checkpoint.get("ema_model_state_dict", {})
    if not ema_state:
        return False

    adapter_state = {}
    prefix = "ema_model."
    for key, value in ema_state.items():
        if not key.startswith(prefix):
            continue
        stripped = key[len(prefix) :]
        if stripped.endswith(".lora_A") or stripped.endswith(".lora_B"):
            adapter_state[stripped] = value.detach().cpu().contiguous()

    if not adapter_state:
        return False

    output_dir.mkdir(parents=True, exist_ok=True)
    save_file(adapter_state, str(output_dir / "adapter_model.safetensors"))
    with open(output_dir / "adapter_config.json", "w", encoding="utf-8") as f:
        json.dump({"format_version": 1, **adapter_config}, f, indent=2, ensure_ascii=False)
    return True


def load_anchor_texts(anchor_text_file: str):
    if not anchor_text_file:
        return []
    path = Path(anchor_text_file).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Missing --distill_anchor_text_file: {path}")
    texts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                texts.append(line)
    if not texts:
        raise ValueError(f"No usable anchor texts found in: {path}")
    return texts


def main():
    args = parse_args()
    dataset_path = Path(args.dataset_path).expanduser().resolve()
    dataset_vocab_file = dataset_path / "vocab.txt"
    if not dataset_vocab_file.exists():
        raise FileNotFoundError(f"Missing vocab file in dataset_path: {dataset_vocab_file}")

    if args.vocab_file:
        vocab_file = Path(args.vocab_file).expanduser().resolve()
        if not vocab_file.exists():
            raise FileNotFoundError(f"Missing provided --vocab_file: {vocab_file}")
    else:
        vocab_file = dataset_vocab_file

    print(f"Using tokenizer vocab: {vocab_file}")
    if dataset_vocab_file != vocab_file:
        print(f"Dataset vocab at {dataset_vocab_file} is not used (override via --vocab_file).")

    model_cfg = OmegaConf.load(str(files("f5_tts").joinpath(f"configs/{args.exp_name}.yaml")))
    model_cls = get_class(f"f5_tts.model.{model_cfg.model.backbone}")
    model_arc = model_cfg.model.arch
    mel_spec_kwargs = model_cfg.model.mel_spec

    vocab_char_map, vocab_size = get_tokenizer(str(vocab_file), tokenizer="custom")
    model = CFM(
        transformer=model_cls(**model_arc, text_num_embeds=vocab_size, mel_dim=mel_spec_kwargs.n_mel_channels),
        mel_spec_kwargs=mel_spec_kwargs,
        vocab_char_map=vocab_char_map,
    )

    ckpt_path = resolve_ckpt_path(args.base_ckpt, args.exp_name)
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "xpu"
        if torch.xpu.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    # Training path expects fp32 model params unless explicit mixed-precision training is wired in.
    # Do not inherit inference auto-fp16 loading here.
    model = load_checkpoint(model, ckpt_path, device=device, dtype=torch.float32, use_ema=args.use_ema)

    freeze_model_parameters(model)
    peft_cfg = PVCAdapterConfig(
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        prompt_drop_path=args.prompt_drop_path,
    )
    injected = apply_pvc_adapters(model, peft_cfg)
    total_params, trainable_params = count_parameters(model)
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,} ({100.0 * trainable_params / total_params:.4f}%)")

    logger = None if args.logger == "none" else args.logger
    distill_anchor_texts = load_anchor_texts(args.distill_anchor_text_file)
    if args.distill_weight > 0 and distill_anchor_texts:
        print(
            "Distillation enabled: "
            f"weight={args.distill_weight}, prob={args.distill_prob}, anchors={len(distill_anchor_texts)}"
        )
    elif args.distill_weight > 0:
        print("Distillation requested but anchor text file is empty. Distillation will be disabled.")

    checkpoint_dir = args.checkpoint_dir or str(files("f5_tts").joinpath(f"../../ckpts/pvc_{args.speaker_id}"))
    periodic_adapter_root = Path(checkpoint_dir).expanduser().resolve() / "lora_adapters"
    periodic_exported_updates = set()

    def on_checkpoint_saved(trainer_obj, update, last, checkpoint_file):
        # Export adapters on periodic "last" saves (default every 1000 updates).
        if not last:
            return
        if update in periodic_exported_updates:
            return
        model_unwrapped_local = trainer_obj.accelerator.unwrap_model(trainer_obj.model)
        adapter_cfg = build_adapter_metadata(
            args=args,
            ckpt_path=ckpt_path,
            vocab_file=str(vocab_file),
            dataset_path=dataset_path,
            peft_cfg=peft_cfg,
            injected=injected,
            total_params=total_params,
            trainable_params=trainable_params,
            mel_spec_kwargs=mel_spec_kwargs,
            checkpoint_update=update,
            checkpoint_file=checkpoint_file,
        )
        adapter_dir = periodic_adapter_root / f"update_{update}"
        exported = save_adapter_from_ema_checkpoint(checkpoint_file, adapter_dir, adapter_cfg)
        if not exported:
            save_adapter(model_unwrapped_local, str(adapter_dir), adapter_cfg)
            print(f"Saved periodic LoRA adapter (online model) at update {update}: {adapter_dir}")
        else:
            print(f"Saved periodic LoRA adapter (EMA) at update {update}: {adapter_dir}")
        periodic_exported_updates.add(update)

    trainer = Trainer(
        model=model,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        num_warmup_updates=args.num_warmup_updates,
        save_per_updates=args.save_per_updates,
        keep_last_n_checkpoints=args.keep_last_n_checkpoints,
        checkpoint_path=checkpoint_dir,
        batch_size_per_gpu=args.batch_size_per_gpu,
        batch_size_type=args.batch_size_type,
        max_samples=args.max_samples,
        grad_accumulation_steps=args.grad_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        logger=logger,
        wandb_project=f"pvc_{args.exp_name}",
        wandb_run_name=f"{args.speaker_id}_pvc_lora",
        last_per_updates=args.last_per_updates,
        mel_spec_type=mel_spec_kwargs.mel_spec_type,
        checkpoint_callback=on_checkpoint_saved,
        distill_weight=args.distill_weight,
        distill_prob=args.distill_prob,
        distill_anchor_texts=distill_anchor_texts,
        distill_student_lora_strength=args.distill_student_lora_strength,
    )

    train_dataset = load_dataset(
        dataset_name=str(dataset_path),
        tokenizer="custom",
        dataset_type="CustomDatasetPath",
        mel_spec_kwargs=mel_spec_kwargs,
    )
    trainer.train(train_dataset, num_workers=args.num_workers, resumable_with_seed=args.seed)

    trainer.accelerator.wait_for_everyone()
    if trainer.is_main:
        output_dir = Path(args.output_root).expanduser().resolve() / args.speaker_id
        model_unwrapped = trainer.accelerator.unwrap_model(trainer.model)
        adapter_cfg = build_adapter_metadata(
            args=args,
            ckpt_path=ckpt_path,
            vocab_file=str(vocab_file),
            dataset_path=dataset_path,
            peft_cfg=peft_cfg,
            injected=injected,
            total_params=total_params,
            trainable_params=trainable_params,
            mel_spec_kwargs=mel_spec_kwargs,
        )
        save_adapter(model_unwrapped, str(output_dir), adapter_cfg)
        print(f"Saved adapter artifact at: {output_dir}")


if __name__ == "__main__":
    main()
