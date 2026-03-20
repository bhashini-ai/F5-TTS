# F5-TTS Implementation Reference

This document is a code-level reference for future implementation work in this repository, focused on your main flow:

- Input: short reference audio (`<10s` recommended), optional/explicit reference text, and target text
- Output: synthesized speech for target text that preserves source speaker characteristics

## 1. Repository Map (What matters for inference)

- `README.md`: user-level install + inference entry commands
- `src/f5_tts/api.py`: Python API wrapper class `F5TTS`
- `src/f5_tts/infer/utils_infer.py`: core inference pipeline
- `src/f5_tts/infer/infer_cli.py`: CLI flow (`.toml` + flags, multi-voice tags)
- `src/f5_tts/infer/infer_gradio.py`: Gradio UI and multi-style flow
- `src/f5_tts/model/cfm.py`: flow-matching model sampling (`CFM.sample`)
- `src/f5_tts/model/utils.py`: tokenizer + Chinese pinyin conversion
- `src/f5_tts/configs/F5TTS_v1_Base.yaml`: default base architecture and mel/vocoder settings
- `src/f5_tts/socket_server.py`: streaming socket server based on `infer_batch_process(streaming=True)`

## 2. End-to-End Synthesis Path

Primary path (API/CLI/Gradio all converge here):

1. `preprocess_ref_audio_text(...)`
2. `infer_process(...)`
3. `infer_batch_process(...)`
4. `CFM.sample(...)`
5. Vocoder decode to waveform (`vocos` or `bigvgan`)

Important files:

- `src/f5_tts/infer/utils_infer.py`
- `src/f5_tts/model/cfm.py`

## 3. Input Handling Details (Reference audio + text)

### 3.1 Reference audio preprocessing

`preprocess_ref_audio_text`:

- Reads source audio with `pydub`
- Tries silence-aware clipping
- Hard caps long reference audio to about 12s
- Trims leading/trailing silence
- Appends ~50ms silence tail
- Writes a temporary WAV and returns its path
- Caches preprocessed result by MD5 of original audio bytes (in-memory process cache)

Implication for your workflow:

- Your `<10s` reference audio is within the code's expected range and avoids clipping artifacts.

### 3.2 Reference text behavior

- If `ref_text` is empty, it auto-transcribes with Whisper pipeline:
  - `openai/whisper-large-v3-turbo`
- Transcription is cached by audio hash in-memory
- If custom `ref_text` is provided, it is used directly
- Ensures ending punctuation by appending `. ` or equivalent when missing

## 4. Generation Mechanics

### 4.1 Text chunking

`infer_process` computes dynamic `max_chars` based on:

- reference text byte length
- reference audio duration
- speed parameter

Then `chunk_text` splits target text by punctuation boundaries to avoid overlong single-pass generation.

### 4.2 Duration estimation

In `infer_batch_process` per chunk:

- Converts text to model tokens (`convert_char_to_pinyin` path)
- Computes target mel duration from reference audio/reference text ratio
- Or uses explicit `fix_duration` when provided

### 4.3 Model sampling

`CFM.sample`:

- Runs neural ODE sampling (`torchdiffeq.odeint`) with configurable steps (`nfe_step`)
- Uses classifier-free guidance (`cfg_strength`)
- Supports sway sampling (`sway_sampling_coef`)
- Keeps conditioning on reference audio region and generates the continuation

### 4.4 Vocoder + post-processing

- Mel output is decoded via:
  - `vocos` (default), or
  - `bigvgan`
- Multi-chunk outputs are stitched with optional cross-fade (`cross_fade_duration`)
- Optional silence removal can be applied on saved WAV

## 5. Interfaces You Can Build On

### 5.1 Python API (best integration target)

File: `src/f5_tts/api.py`

- Class: `F5TTS`
- Method: `infer(ref_file, ref_text, gen_text, ...)`
- Returns: `(wav_array, sample_rate, spectrogram)`

Use this for service/API integration because it encapsulates model/vocoder loading and the core pipeline.

### 5.2 CLI

File: `src/f5_tts/infer/infer_cli.py`

- Supports config via `.toml`
- Supports multi-voice scripts with `[voice_tag]...`
- Can save per-chunk wave files

### 5.3 Gradio

File: `src/f5_tts/infer/infer_gradio.py`

- Basic single-reference generation
- Multi-style/multi-speaker style-tag generation
- Includes seed/speed controls and optional silence removal

### 5.4 Streaming (low-latency)

File: `src/f5_tts/socket_server.py`

- Uses `infer_batch_process(streaming=True)` and sends float chunks over TCP
- Good base for realtime/interactive voice services

## 6. Key Tuning Knobs (Quality vs latency)

All exposed in API/CLI pipeline:

- `nfe_step`: denoising steps (higher = slower, often better quality)
- `cfg_strength`: classifier-free guidance strength
- `sway_sampling_coef`: alternative timestep shaping
- `speed`: speaking rate
- `cross_fade_duration`: chunk stitching smoothness
- `fix_duration`: force total duration (ref+generated)
- `remove_silence`: cleanup long pauses

## 7. Current Defaults (from code)

- Default model: `F5TTS_v1_Base`
- Default sample rate: `24000`
- Default mel channels: `100`
- Default vocoder path: `vocos` unless overridden
- Device priority: `cuda -> xpu -> mps -> cpu`

## 8. Extension Points for Future Implementations

### 8.1 Custom checkpoint/model swap

- API: set `model`, `ckpt_file`, `vocab_file`
- CLI: `--model`, `--ckpt_file`, `--vocab_file`
- Ensure model config and mel/vocoder types stay aligned

### 8.2 Better reference-audio policy

Edit: `preprocess_ref_audio_text` in `utils_infer.py`

- Change max length policy (currently ~12s)
- Change silence thresholds/seek steps
- Add loudness normalization or denoise stages

### 8.3 Chunking policy upgrades

Edit: `chunk_text` and `infer_process`

- Add language-aware sentence segmentation
- Introduce semantic chunk boundaries
- Add chunk-level prosody controls

### 8.4 Multilingual/tokenization changes

Edit: `src/f5_tts/model/utils.py`

- `convert_char_to_pinyin`
- `get_tokenizer`

### 8.5 Serviceization

- Start with `F5TTS` API class for stateless service endpoints
- For streaming responses, adapt `socket_server.py` logic into WebSocket/gRPC transport

## 9. Practical Implementation Checklist

For stable voice-clone generation in production-like use:

1. Keep reference audio clean and short (`5-10s` preferred, `<12s` enforced by preprocessing logic).
2. Provide accurate `ref_text` whenever possible (better than ASR fallback).
3. Start with `F5TTS_v1_Base`, `nfe_step=32`, `cfg_strength=2`.
4. Enable chunk cross-fade for long target text.
5. Keep seeds for reproducibility if deterministic outputs are needed.
6. Add output post-processing (silence trim, loudness normalization) at service boundary if needed.

## 10. Minimal API Example

```python
from f5_tts.api import F5TTS

f5 = F5TTS(model="F5TTS_v1_Base")
wav, sr, spec = f5.infer(
    ref_file="path/to/ref.wav",
    ref_text="reference transcript here",
    gen_text="target text to synthesize",
    nfe_step=32,
    cfg_strength=2.0,
    speed=1.0,
    cross_fade_duration=0.15,
    remove_silence=False,
)
```

Use this file as the baseline when adding new features (API wrappers, custom preprocessing, streaming transports, model swaps, and quality/latency tuning).
