# PVC Implementation Plan (LoRA + Prompt Adapter) for F5-TTS

## 1. Target Outcome

Design goal we will optimize for:

- One shared frozen base F5-TTS model
- Tiny per-voice artifacts (MB-scale, not GB-scale)
- Hot-swappable per-request voices in Triton
- No GPU memory explosion
- PVC-like quality from 2-3 hours per speaker

Expected per-speaker artifact:

```text
lora/
 └── speaker_001/
     ├── adapter_model.safetensors
     └── adapter_config.json
```

## 2. Paper-to-Repo Mapping

Paper used: *Parameter-Efficient Fine-Tuning for Low-Resource Text-to-Speech via Cross-Lingual Continual Learning* (Interspeech 2025).

Methods we will adopt from the paper:

- Prompt Adapter: LoRA on the input projection layer after text+audio concat, with DropPath regularization
- DiT LoRA Adapter: LoRA on DiT attention (query + value), rank 16 baseline

Directly relevant paper settings:

- LoRA rank: 16 (primary baseline)
- DiT LoRA on Q and V projections
- LoRA dropout: 0.05
- DropPath: 0.3
- AdamW, LR around `1e-5` in paper setup

For your PVC scenario (same language, per-speaker clone), we do **not** need to start with full text-encoder adaptation. We keep text encoder frozen by default and train only adapter modules.

## 3. Current Repo Touchpoints

### Model/inference

- `src/f5_tts/model/backbones/dit.py`
  - `InputEmbedding.proj` is the correct Prompt Adapter insertion point.
- `src/f5_tts/model/modules.py`
  - `Attention.to_q` / `Attention.to_v` are the DiT LoRA target layers.
- `src/f5_tts/api.py`
  - Main Python API integration point for adapter load/switch.
- `src/f5_tts/infer/utils_infer.py`
  - Core inference path (no adapter support yet).

### Training

- `src/f5_tts/train/finetune_cli.py`
- `src/f5_tts/model/trainer.py`
- `src/f5_tts/model/dataset.py`
- `src/f5_tts/train/datasets/prepare_csv_wavs.py`

### Triton runtime

- `src/f5_tts/runtime/triton_trtllm/model_repo_f5_tts/f5_tts/1/model.py`
- `src/f5_tts/runtime/triton_trtllm/model_repo_f5_tts/f5_tts/config.pbtxt`

## 4. Architecture Decision (Important)

Current Triton path is TensorRT-LLM engine based. True per-request LoRA hot-swap inside this engine path is non-trivial.

Plan:

- Keep existing TRT-LLM deployment for zero-shot baseline traffic.
- Add a **PVC Triton Python backend path** that keeps one frozen PyTorch base model in GPU and applies per-request adapters dynamically.
- Share the same vocoder model repo (`vocoder` / `vocoder_bigvgan`) to avoid duplication.

This gives immediate hot-swap behavior and adapter caching, while avoiding per-speaker full model load.

## 5. Implementation Phases

## Phase A: PEFT module foundation

Deliverables:

- Add new PEFT modules under `src/f5_tts/peft/`:
  - `lora.py`: `LoRALinear` wrapper, merge/unmerge support
  - `prompt_adapter.py`: Prompt adapter wrapper for `InputEmbedding.proj`
  - `drop_path.py`: DropPath module
  - `io.py`: save/load adapter weights + config
  - `inject.py`: utility to inject adapters into a loaded base model

Design:

- Base model parameters frozen.
- Trainable parameters:
  - Prompt adapter LoRA (`input_embed.proj`)
  - DiT LoRA (`to_q`, `to_v` in each transformer block)
- Optional (off by default): text encoder adapter branch for future language adaptation.

Acceptance:

- Without adapter: output matches current model behavior.
- With adapter loaded: model runs and outputs differ deterministically by adapter.

## Phase B: PVC fine-tuning pipeline (per speaker)

Deliverables:

- New training entrypoint:
  - `src/f5_tts/train/pvc_finetune_cli.py`
- Speaker-centric config:
  - `src/f5_tts/configs/pvc_lora_prompt.yaml`
- Optional helper:
  - `scripts/pvc_train_all_speakers.sh` (train one adapter per speaker ID)

Training data assumptions (per speaker):

- 2-3 hours total
- 3-12s sentence-level clips preferred
- Rich punctuation/emotion/speaking-rate diversity

Recommended initial hyperparams:

- LoRA rank: 16
- LoRA alpha: 16
- LoRA dropout: 0.05
- Prompt DropPath: 0.3
- LR: `1e-5` to `2e-5`
- Batch policy: existing frame-based dynamic batching
- Gradient clip: 1.0

Artifact export:

- Save only adapter weights + config:
  - `adapter_model.safetensors`
  - `adapter_config.json`
- Include metadata in config:
  - `base_model_name`, `base_checkpoint_sha`, `rank`, `target_modules`, `sample_rate`, `speaker_id`

Acceptance:

- Per-speaker adapter directory is produced in the required format.
- Adapter size remains MB-scale.

## Phase C: Inference API/CLI integration

Deliverables:

- Extend `src/f5_tts/api.py`:
  - `load_adapter(adapter_dir)`
  - `unload_adapter()`
  - `infer(..., speaker_adapter=None)` (optional override per call)
- Extend `src/f5_tts/infer/infer_cli.py`:
  - `--adapter_dir` flag
  - config TOML key `adapter_dir`

Acceptance:

- Same base model can synthesize with different speaker adapters without restart.
- Switching adapter does not reload base weights.

## Phase D: Triton hot-swap PVC runtime

Deliverables:

- New model repo (recommended):
  - `src/f5_tts/runtime/triton_pvc/model_repo_f5_tts_pvc/f5_tts_pvc/`
- Extend Triton model config inputs:
  - `speaker_id` (string) or `adapter_id` (string)
- Implement `AdapterManager`:
  - LRU cache of top-N adapters on GPU
  - CPU cache for inactive adapters
  - Eviction policy by memory budget and LRU

Runtime flow:

```text
Request
  -> resolve speaker_id -> adapter path
  -> if adapter in GPU cache: activate
  -> else load safetensors to CPU, move to GPU, activate
  -> run inference with frozen base + active adapter
  -> update LRU
```

Batching strategy:

- Group requests by `speaker_id` inside `execute()` to minimize adapter thrash.
- If mixed adapters in same Triton batch, process in sub-batches by adapter key.

Acceptance:

- Base model stays loaded once.
- Adapter switching works per request.
- Cache hit path avoids adapter reload overhead.

## Phase E: Evaluation and gating

Objective metrics (automated):

- CER/WER (ASR transcription)
- Speaker similarity (ECAPA/WavLM cosine)
- UTMOS / MOS proxy

Regression checks:

- Zero-shot quality unchanged when no adapter is selected.
- PVC adapter significantly improves speaker similarity and naturalness vs zero-shot reference prompting.

Operational checks:

- Adapter cache memory remains bounded.
- No growth in GPU memory after repeated adapter switches (leak check).

## 6. GPU Memory Strategy

- Keep one frozen base model loaded once.
- Keep only top-N adapters on GPU (`N` configurable, e.g., 8-32 depending on GPU RAM).
- Keep remaining adapters in CPU RAM or disk; lazy-load on demand.
- Do not clone full model per speaker.

Estimated adapter size for rank-16 Q/V LoRA + prompt adapter is typically a few MB in fp16.

## 7. Proposed Request/Storage Contracts

### Triton request additions

- `speaker_id` (required for PVC route)
- `adapter_revision` (optional, for rollback/version pinning)

### Adapter registry

- Maintain registry file (JSON/YAML) mapping `speaker_id -> adapter_dir`.
- Validate IDs to prevent path traversal.
- Store base-model compatibility metadata in `adapter_config.json`.

## 8. Risks and Mitigations

- Risk: Adapter overfitting to narrow speaking style.
  - Mitigation: enforce dataset diversity, apply DropPath, early stopping by validation similarity + CER.
- Risk: Adapter switching overhead under high QPS.
  - Mitigation: LRU cache + speaker-grouped sub-batching.
- Risk: Incompatibility across base model updates.
  - Mitigation: strict base checkpoint hash in adapter config; reject mismatched adapters.
- Risk: Current TRT-LLM path cannot directly hot-swap LoRA.
  - Mitigation: separate PVC Python backend route while retaining TRT path for zero-shot.

## 9. Milestone Plan

1. Milestone 1 (PEFT core + training): adapter modules + per-speaker training + artifact export.
2. Milestone 2 (offline inference): API/CLI load/switch adapters locally.
3. Milestone 3 (Triton PVC route): per-request `speaker_id`, adapter cache, production benchmark.
4. Milestone 4 (hardening): eval dashboard, compatibility checks, failure handling.

## 10. Definition of Done

- You can train one adapter per speaker from 2-3 hours data.
- You store only per-speaker adapter artifacts in MB scale.
- Triton serves multiple speakers by hot-swapping adapters on one shared base model.
- Quality is clearly above zero-shot cloning on internal evaluation set.
