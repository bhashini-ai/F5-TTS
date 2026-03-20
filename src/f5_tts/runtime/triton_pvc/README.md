## Triton PVC Runtime (Hot-Swappable LoRA Adapters)

This runtime provides a Python-backend Triton route for PVC inference with:

- one shared frozen base F5-TTS model per GPU instance
- per-request adapter switching by `speaker_id` / `adapter_id`
- GPU LRU cache for top-N adapters

### Model Repo

Use:

- `src/f5_tts/runtime/triton_pvc/model_repo_f5_tts_pvc/f5_tts_pvc/config.pbtxt`
- `src/f5_tts/runtime/triton_pvc/model_repo_f5_tts_pvc/f5_tts_pvc/1/model.py`

### Required Inputs

- `reference_wav` (FP32)
- `reference_text` (STRING)
- `target_text` (STRING)

### Adapter Inputs

- `speaker_id` (STRING, optional)
- `adapter_id` (STRING, optional)
- `adapter_revision` (STRING, optional)

Resolution order:

1. if `adapter_id` present, use it
2. else use `speaker_id`
3. if neither present, run base model without adapter

### Adapter Path Resolution

Configure via `config.pbtxt` parameters:

- `lora_root`: base directory where adapters are stored as `lora/<speaker_or_adapter_id>/...`
- `adapter_registry`: optional JSON mapping ID -> absolute adapter directory
- adapter IDs/revisions are resolved with path-safety checks; traversal outside configured roots is rejected

### Adapter Cache

- `adapter_cache_size_gpu`: top-N adapters cached in GPU memory
- `adapter_cache_size_cpu`: additional adapters cached in CPU memory
- grouped execution by adapter key minimizes adapter thrash within dynamic batches
- periodic cache metrics logging is controlled by `log_metrics_every_n_requests` in `config.pbtxt`

### Compatibility Checks

At adapter activation time, the runtime validates adapter compatibility (strict by default):

- adapter model name (`exp_name` / `model_name`) vs runtime base model
- adapter `base_ckpt_sha256` vs runtime base checkpoint hash
- in strict mode, missing metadata is also treated as incompatibility

If mismatched and `strict_adapter=true`, request fails fast.
If `adapter_revision` is provided and missing on disk, request fails fast.

### Example Request Fields (HTTP v2 payload)

- `reference_wav`
- `reference_wav_len` (optional)
- `reference_text`
- `target_text`
- `speaker_id` (e.g. `speaker_001`)

### Notes

- This route is separate from TensorRT-LLM runtime and is intended for hot-swappable PVC adapters.
- Per-speaker artifacts are expected from Milestone 1:
  - `adapter_model.safetensors`
  - `adapter_config.json`
