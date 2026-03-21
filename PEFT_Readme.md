# PEFT README: Professional Voice Cloning (PVC) in F5-TTS

This guide explains how to train and serve **per-speaker LoRA adapters** for professional voice cloning (PVC) in this repo.

Goal:
- Keep one shared frozen base F5-TTS model
- Train tiny per-speaker adapters (MB-scale)
- Hot-swap speakers at inference time

---

## 1) Quick Mental Model

You do **not** fine-tune the full model per speaker.

You:
1. Prepare one speaker dataset (2-3 hours, sentence-level clips).
2. Run PVC PEFT fine-tuning to train only adapter params.
3. Save artifacts:
   - `adapter_model.safetensors`
   - `adapter_config.json`
4. Load adapter at inference time (CLI/API/Triton), while base model stays shared.

---

## 2) Environment Prerequisites

- Python environment with repo dependencies installed
- CUDA GPU strongly recommended
- `ffmpeg` available (recommended for robust duration probing)

Check ffmpeg:

```bash
ffmpeg -version
```

Install repo in editable mode from repo root (recommended):

```bash
pip install -e .
```

---

## 3) Training Data Preparation (Exact Format)

PVC training uses the same prepared dataset format expected by `CustomDatasetPath`:
- `raw.arrow`
- `duration.json`
- `vocab.txt`

You generate these from a CSV.

### 3.1 CSV format (strict)

Create `metadata.csv` with `|` delimiter and header exactly:

```text
audio_file|text
/abs/path/to/speaker_001/0001.wav|Hello, this is speaker one.
/abs/path/to/speaker_001/0002.wav|Today we are testing punctuation, speed, and style.
```

Rules:
- `audio_file` must be **absolute path** (required by loader)
- One utterance per row
- Sentence-level clips preferred: ~3s to 12s
- Keep transcription clean and accurate
- Collect diverse speaking styles (emotion, pace, punctuation)

### 3.2 Convert CSV -> train-ready dataset

```bash
python src/f5_tts/train/datasets/prepare_csv_wavs.py /abs/path/metadata.csv /abs/path/data/speaker_001_pvc
```

Expected output directory:

```text
/abs/path/data/speaker_001_pvc/
  raw.arrow
  duration.json
  vocab.txt
```

Note:
- For fine-tuning path, script copies pretrained vocab into `vocab.txt`.

---

## 4) Train PVC LoRA Adapter

Run per speaker:

```bash
f5-tts_pvc-finetune-cli \
  --exp_name F5TTS_v1_Base \
  --dataset_path /abs/path/data/speaker_001_pvc \
  --speaker_id speaker_001 \
  --lora_rank 16 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --prompt_drop_path 0.3 \
  --conditioning_adapter \
  --conditioning_gamma 0.25 \
  --conditioning_kernel_size 3 \
  --conditioning_se_reduction 4 \
  --learning_rate 1e-5 \
  --batch_size_per_gpu 3200 \
  --batch_size_type frame \
  --epochs 100 \
  --output_root /abs/path/lora
```

Optional knobs:
- `--base_ckpt /abs/path/model_1250000.safetensors` to pin a local base checkpoint
- `--use_ema/--no-use_ema`
- `--checkpoint_dir /abs/path/ckpts/pvc_speaker_001`
- `--no-conditioning_adapter` to disable conditioning adapter for legacy compatibility experiments

Output artifacts:

```text
/abs/path/lora/speaker_001/
  adapter_model.safetensors
  adapter_config.json
```

The config includes compatibility metadata (`exp_name`, `base_ckpt_sha256`, PEFT params), used by inference hardening.

---

## 5) Use Fine-Tuned Adapter from Command Line

### 5.1 One-shot CLI inference with adapter

```bash
f5-tts_infer-cli \
  --model F5TTS_v1_Base \
  --adapter_dir /abs/path/lora/speaker_001 \
  --ref_audio /abs/path/ref.wav \
  --ref_text "Reference transcript here." \
  --gen_text "This is the target sentence to synthesize."
```

You can also set `adapter_dir` in TOML config used by `f5-tts_infer-cli -c ...`.

### 5.2 Compatibility checks at load time

When adapter is loaded, runtime validates:
- adapter model name vs current base model
- adapter `base_ckpt_sha256` vs runtime checkpoint hash

If strict mode is on and metadata mismatches (or missing), load fails fast.

---

## 6) Multilingual Retention Features (New)

You now have two practical controls to keep PVC quality high while reducing cross-lingual regression.

### 6.1 Feature A: `lora_strength` at inference time

`lora_strength` controls how strongly the speaker adapter influences generation:
- `0.0` = base model behavior (adapter effectively off)
- `1.0` = full adapter behavior
- between `0.0` and `1.0` = blend

Use it when:
- PVC voice similarity is excellent in speaker language
- but Hindi (or another language) pronunciation/prosody regresses

Example:

```bash
f5-tts_infer-cli \
  --model F5TTS_v1_Base \
  --ckpt_file /abs/path/model_remapped.safetensors \
  --vocab_file /abs/path/vocab.txt \
  --adapter_dir /abs/path/lora/speaker_001 \
  --lora_strength 0.75 \
  --model_dtype float32 \
  --ref_audio /abs/path/ref_kn.wav \
  --ref_text "ಕನ್ನಡ ಉಲ್ಲೇಖ ವಾಕ್ಯ." \
  --gen_text "हिंदी लक्ष्य वाक्य।"
```

If you hit `NaN/Inf` mel errors, first try `--model_dtype float32`.

### 6.2 Feature B: Distillation anchors during PVC training

Distillation anchors regularize LoRA so it keeps more base-model multilingual behavior.

How it works (high level):
1. For some updates, sample anchor text lines.
2. Run base behavior with LoRA temporarily set to `0.0`.
3. Run student behavior with LoRA active.
4. Add an MSE distillation loss so LoRA does not drift too far from base multilingual priors.

Training command:

```bash
f5-tts_pvc-finetune-cli \
  --exp_name F5TTS_v1_Base \
  --base_ckpt /abs/path/model_remapped.safetensors \
  --dataset_path /abs/path/data/speaker_001_pvc \
  --speaker_id speaker_001 \
  --output_root /abs/path/lora \
  --distill_weight 0.1 \
  --distill_prob 0.3 \
  --distill_anchor_text_file /abs/path/distill_anchor_texts_kn_hi.txt \
  --distill_student_lora_strength 1.0
```

Example `distill_anchor_text_file` (`UTF-8`, one line per anchor):

```text
ಇದು ಕನ್ನಡದಲ್ಲಿ ಸರಳ ಘೋಷಣಾ ವಾಕ್ಯವಾಗಿದೆ.
ಸ್ವರ, ವ್ಯಂಜನ, ವಿರಾಮ ಚಿಹ್ನೆಗಳೊಂದಿಗೆ ಓದಿ.
आज मौसम साफ है और यातायात सामान्य है।
कृपया अगले स्टेशन पर उतरने की तैयारी करें।
Invoice number 4521 was generated on 14-02-2026.
ಬೆಂಗಳೂರು से दिल्ली की उड़ान शाम को प्रस्थान करेगी।
```

Guidelines for effective anchors:
- Use 200-2000 lines (start with ~500).
- Mix languages you want to preserve (for your case: Kannada + Hindi).
- Include punctuation, numerals, dates, abbreviations, and medium-length sentences.
- Avoid speaker-specific private content; use generic text.
- Keep transcription-normalized script (no random spelling variants).

Starter hyperparameters:
- `distill_weight`: `0.05` to `0.2` (start `0.1`)
- `distill_prob`: `0.2` to `0.5` (start `0.3`)
- `distill_student_lora_strength`: `1.0`

What difference to expect:
- Kannada speaker similarity may drop slightly if distillation is too strong.
- Hindi intelligibility/prosody usually improves vs pure single-language LoRA.
- Net result is typically better bilingual balance, not absolute max in one language.

### 6.3 How to test quality properly

Use a fixed bilingual evaluation set:
- 30-50 Kannada target sentences
- 30-50 Hindi target sentences
- same reference audio style across experiments

For each checkpoint and strength:
1. Generate the same fixed Kannada/Hindi eval set with `lora_strength` in `{0.0, 0.25, 0.5, 0.75, 1.0}`.
2. Compute CER/WER with your preferred local ASR pipeline (or run human transcription checks if GPU budget is tight).
3. Do listening checks on 10-20 hardest Hindi lines (numbers, names, punctuation).
4. Select pair: `(checkpoint, lora_strength)` with best objective and acceptable voice similarity.

---

## 7) What Happens Internally (Training + Inference)

### 7.1 Training internals (PVC fine-tune)

1. Load frozen base checkpoint (`F5TTS_v1_Base` by default).
2. Freeze base model parameters.
3. Inject PEFT modules:
   - Prompt adapter LoRA at input projection
   - DiT LoRA on attention targets (Q/V via configured regex)
   - Conditioning adapter on ConvNeXtV2 text-embedding depth-wise conv layers (`text_embed.text_blocks.*.dwconv`)
4. Train only adapter params with existing Trainer loop.
5. Export only LoRA weights + config metadata.

Result: MB-scale speaker artifact, not full-model copy.

### 7.2 Inference internals (local CLI/API)

1. Load shared base model once.
2. Inject adapter module structure once.
3. Load adapter weights into LoRA modules.
4. Generate audio from reference + target text.

Switching speaker means switching adapter weights, not reloading base model.

---

## 8) Deploy PVC to Triton Inference Server

PVC runtime is a separate Triton Python backend route:
- `src/f5_tts/runtime/triton_pvc/model_repo_f5_tts_pvc/f5_tts_pvc`

### 8.1 Expected serving layout

```text
/models/
  model_repo_f5_tts_pvc/
    f5_tts_pvc/
      config.pbtxt
      1/model.py
  lora/
    speaker_001/
      adapter_model.safetensors
      adapter_config.json
    speaker_002/
      adapter_model.safetensors
      adapter_config.json
```

### 8.2 Configure `config.pbtxt`

Key parameters:
- `model_name`
- `ckpt_file` (optional; default HF path if empty)
- `lora_root` (e.g. `/models/lora`)
- `adapter_registry` (optional JSON map)
- `adapter_cache_size_gpu`
- `adapter_cache_size_cpu`
- `strict_adapter`
- `log_metrics_every_n_requests`
- `conditioning_enabled` / `conditioning_gamma` / `conditioning_kernel_size` / `conditioning_se_reduction`

### 8.3 Start Triton (example)

Use a Triton image with Python backend, mount model repo and adapter dir.
Ensure `f5_tts` Python package + dependencies are installed in server environment.

```bash
docker run --rm --gpus all --net host \
  -v /abs/path/models/model_repo_f5_tts_pvc:/models/model_repo_f5_tts_pvc \
  -v /abs/path/models/lora:/models/lora \
  nvcr.io/nvidia/tritonserver:24.12-py3 \
  tritonserver --model-repository=/models/model_repo_f5_tts_pvc
```

If your container does not already include this repo as importable package, install it in the container:

```bash
pip install -e /abs/path/F5-TTS-BhashiniAI
```

### 8.4 Triton PVC runtime flow

Per request:
1. Parse `speaker_id` or `adapter_id` (+ optional `adapter_revision`).
2. Resolve adapter directory (with path safety checks).
3. Check compatibility metadata.
4. Load from GPU cache, CPU cache, or disk.
5. Apply adapter to shared base model.
6. Run synthesis.
7. Update LRU metrics and optionally log cache stats.

---

## 9) Call Triton from gRPC Client

Inputs expected by `f5_tts_pvc`:
- `reference_wav` (FP32)
- `reference_wav_len` (INT32, optional)
- `reference_text` (STRING)
- `target_text` (STRING)
- `speaker_id` (STRING, optional)
- `adapter_id` (STRING, optional)
- `adapter_revision` (STRING, optional)

Minimal Python gRPC example:

```python
import numpy as np
import soundfile as sf
import tritonclient.grpc as grpcclient
from tritonclient.utils import np_to_triton_dtype

audio, sr = sf.read("/abs/path/ref.wav")
if audio.ndim > 1:
    audio = audio.mean(axis=1)
audio = audio.astype(np.float32)
audio = audio.reshape(1, -1)  # batch=1
audio_len = np.array([[audio.shape[1]]], dtype=np.int32)

ref_text = np.array([["Reference transcript here."]], dtype=object)
target_text = np.array([["Generate this with speaker_001 voice."]], dtype=object)
speaker_id = np.array([["speaker_001"]], dtype=object)

inputs = []

inp = grpcclient.InferInput("reference_wav", audio.shape, np_to_triton_dtype(audio.dtype))
inp.set_data_from_numpy(audio)
inputs.append(inp)

inp = grpcclient.InferInput("reference_wav_len", audio_len.shape, np_to_triton_dtype(audio_len.dtype))
inp.set_data_from_numpy(audio_len)
inputs.append(inp)

inp = grpcclient.InferInput("reference_text", ref_text.shape, "BYTES")
inp.set_data_from_numpy(ref_text)
inputs.append(inp)

inp = grpcclient.InferInput("target_text", target_text.shape, "BYTES")
inp.set_data_from_numpy(target_text)
inputs.append(inp)

inp = grpcclient.InferInput("speaker_id", speaker_id.shape, "BYTES")
inp.set_data_from_numpy(speaker_id)
inputs.append(inp)

client = grpcclient.InferenceServerClient(url="localhost:8001")
result = client.infer(model_name="f5_tts_pvc", inputs=inputs)
waveform = result.as_numpy("waveform").astype(np.float32)
sf.write("pvc_out.wav", waveform, 24000)
```

---

## 10) Operational Notes and Troubleshooting

- Keep adapter and base checkpoint aligned. If base checkpoint changes, retrain or re-export adapters.
- If strict compatibility fails, inspect `adapter_config.json` fields:
  - `exp_name` / `model_name`
  - `base_ckpt_sha256`
- If adapter is not found:
  - verify `lora_root` path in `config.pbtxt`
  - verify adapter folder name matches `speaker_id` or `adapter_id`
- If `adapter_revision` is provided, corresponding subdirectory must exist.
- `reference_text` must be non-empty in PVC Triton runtime.

---

## 11) End-to-End Command Sequence (Copy/Paste)

```bash
# 1) Prepare dataset from CSV
python src/f5_tts/train/datasets/prepare_csv_wavs.py \
  /abs/path/metadata.csv \
  /abs/path/data/speaker_001_pvc

# 2) Train speaker LoRA adapter
f5-tts_pvc-finetune-cli \
  --exp_name F5TTS_v1_Base \
  --dataset_path /abs/path/data/speaker_001_pvc \
  --speaker_id speaker_001 \
  --output_root /abs/path/lora

# 3) Local CLI inference with that adapter
f5-tts_infer-cli \
  --model F5TTS_v1_Base \
  --adapter_dir /abs/path/lora/speaker_001 \
  --ref_audio /abs/path/ref.wav \
  --ref_text "Reference transcript." \
  --gen_text "Target text to synthesize."
```

This is the minimum PVC loop: **prepare -> train adapter -> infer with adapter**.
