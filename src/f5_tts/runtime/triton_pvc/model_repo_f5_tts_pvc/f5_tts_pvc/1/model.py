from __future__ import annotations

import json
import os
import threading
from collections import OrderedDict
from importlib.resources import files

import numpy as np
import torch
import torchaudio
import triton_python_backend_utils as pb_utils
from cached_path import cached_path
from hydra.utils import get_class
from omegaconf import OmegaConf
from torch.utils.dlpack import from_dlpack

from f5_tts.infer.utils_infer import chunk_text, infer_batch_process, load_model, load_vocoder
from f5_tts.peft import (
    PVCAdapterConfig,
    apply_adapter_state,
    apply_pvc_adapters,
    check_adapter_compatibility,
    clear_adapter,
    file_sha256,
    load_adapter_state,
)


def _decode_bool(value: str, default: bool = False) -> bool:
    if value is None:
        return default
    value = str(value).strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return default


def _decode_optional_float(value: str):
    value = str(value).strip()
    if not value:
        return None
    return float(value)


class AdapterManager:
    def __init__(
        self,
        model,
        device: str,
        lora_root: str,
        adapter_registry: dict[str, str] | None = None,
        gpu_cache_size: int = 8,
        cpu_cache_size: int = 64,
        strict: bool = True,
        expected_model_name: str = "",
        expected_base_ckpt_sha256: str = "",
    ):
        self.model = model
        self.device = device
        self.lora_root = os.path.abspath(lora_root) if lora_root else ""
        self.adapter_registry = adapter_registry or {}
        self.gpu_cache_size = max(0, gpu_cache_size)
        self.cpu_cache_size = max(0, cpu_cache_size)
        self.strict = strict
        self.expected_model_name = expected_model_name
        self.expected_base_ckpt_sha256 = expected_base_ckpt_sha256

        self.gpu_cache = OrderedDict()
        self.cpu_cache = OrderedDict()
        self.active_key = None
        self.lock = threading.Lock()
        self.metrics = {
            "activations": 0,
            "base_activations": 0,
            "activation_failures": 0,
            "gpu_cache_hits": 0,
            "cpu_cache_hits": 0,
            "disk_cache_misses": 0,
            "gpu_cache_evictions": 0,
            "cpu_cache_evictions": 0,
        }

    def _safe_resolve_under_root(self, root: str, relative_id: str):
        candidate = os.path.abspath(os.path.join(root, relative_id))
        root_norm = os.path.abspath(root)
        root_prefix = root_norm if root_norm.endswith(os.sep) else root_norm + os.sep
        if candidate != root_norm and not candidate.startswith(root_prefix):
            raise RuntimeError(f"Adapter id resolves outside lora_root: '{relative_id}'")
        return candidate

    def _lru_get(self, cache: OrderedDict, key):
        if key not in cache:
            return None
        value = cache.pop(key)
        cache[key] = value
        return value

    def _lru_put(self, cache: OrderedDict, key, value, max_items: int, cache_name: str):
        if max_items <= 0:
            return
        if key in cache:
            cache.pop(key)
        cache[key] = value
        while len(cache) > max_items:
            cache.popitem(last=False)
            if cache_name == "gpu":
                self.metrics["gpu_cache_evictions"] += 1
            elif cache_name == "cpu":
                self.metrics["cpu_cache_evictions"] += 1

    def _resolve_adapter(self, speaker_id: str, adapter_id: str, adapter_revision: str):
        lookup = adapter_id or speaker_id
        if not lookup:
            return None, None

        if lookup in self.adapter_registry:
            adapter_dir = os.path.abspath(self.adapter_registry[lookup])
        elif self.lora_root:
            adapter_dir = self._safe_resolve_under_root(self.lora_root, lookup)
        else:
            raise RuntimeError("No lora_root or adapter_registry set, cannot resolve adapter path.")

        if adapter_revision:
            revision_path = self._safe_resolve_under_root(adapter_dir, adapter_revision)
            if os.path.isdir(revision_path):
                adapter_dir = revision_path
                cache_key = f"{lookup}:{adapter_revision}"
            else:
                raise FileNotFoundError(
                    f"Requested adapter_revision '{adapter_revision}' not found under: {adapter_dir}"
                )
        else:
            cache_key = lookup

        adapter_dir = os.path.abspath(adapter_dir)
        if not os.path.isdir(adapter_dir):
            raise FileNotFoundError(f"Adapter directory not found: {adapter_dir}")
        return cache_key, adapter_dir

    def _to_gpu_state(self, cpu_state: dict):
        return {k: v.to(device=self.device, non_blocking=True) for k, v in cpu_state.items()}

    def activate(self, speaker_id: str = "", adapter_id: str = "", adapter_revision: str = ""):
        with self.lock:
            self.metrics["activations"] += 1
            try:
                cache_key, adapter_dir = self._resolve_adapter(speaker_id, adapter_id, adapter_revision)
                if cache_key is None:
                    clear_adapter(self.model)
                    self.active_key = None
                    self.metrics["base_activations"] += 1
                    return {"adapter_key": None, "source": "base"}

                if self.active_key == cache_key:
                    return {"adapter_key": cache_key, "source": "already_active"}

                gpu_entry = self._lru_get(self.gpu_cache, cache_key)
                if gpu_entry is not None:
                    apply_adapter_state(self.model, gpu_entry["state"], config=gpu_entry["config"], strict=self.strict)
                    self.active_key = cache_key
                    self.metrics["gpu_cache_hits"] += 1
                    return {"adapter_key": cache_key, "source": "gpu_cache"}

                cpu_entry = self._lru_get(self.cpu_cache, cache_key)
                if cpu_entry is None:
                    state_cpu, config = load_adapter_state(adapter_dir, device="cpu")
                    check_adapter_compatibility(
                        config,
                        expected_model_name=self.expected_model_name,
                        expected_base_ckpt_sha256=self.expected_base_ckpt_sha256,
                        strict=self.strict,
                    )
                    cpu_entry = {
                        "state": state_cpu,
                        "config": config,
                        "adapter_dir": adapter_dir,
                    }
                    self._lru_put(self.cpu_cache, cache_key, cpu_entry, self.cpu_cache_size, cache_name="cpu")
                    self.metrics["disk_cache_misses"] += 1
                else:
                    self.metrics["cpu_cache_hits"] += 1

                state_gpu = self._to_gpu_state(cpu_entry["state"])
                gpu_entry = {"state": state_gpu, "config": cpu_entry["config"], "adapter_dir": adapter_dir}
                self._lru_put(self.gpu_cache, cache_key, gpu_entry, self.gpu_cache_size, cache_name="gpu")

                apply_adapter_state(self.model, state_gpu, config=gpu_entry["config"], strict=self.strict)
                self.active_key = cache_key
                return {"adapter_key": cache_key, "source": "cpu_or_disk"}
            except Exception:
                self.metrics["activation_failures"] += 1
                raise

    def get_metrics(self):
        return {
            **self.metrics,
            "gpu_cache_size": len(self.gpu_cache),
            "cpu_cache_size": len(self.cpu_cache),
            "active_adapter_key": self.active_key,
        }


class TritonPythonModel:
    def initialize(self, args):
        model_config = json.loads(args["model_config"])
        parameters = model_config.get("parameters", {})
        self.parameters = {k: v.get("string_value", "") for k, v in parameters.items()}
        model_dir = pb_utils.get_model_dir()

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "xpu"
            if torch.xpu.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )

        model_name = self.parameters.get("model_name", "F5TTS_v1_Base")
        model_cfg = OmegaConf.load(str(files("f5_tts").joinpath(f"configs/{model_name}.yaml")))
        model_cls = get_class(f"f5_tts.model.{model_cfg.model.backbone}")
        model_arc = model_cfg.model.arch

        self.reference_sample_rate = int(self.parameters.get("reference_audio_sample_rate", "24000"))
        self.target_rms = float(self.parameters.get("target_rms", "0.1"))
        self.nfe_step = int(self.parameters.get("nfe_step", "32"))
        self.cfg_strength = float(self.parameters.get("cfg_strength", "2.0"))
        self.sway_sampling_coef = float(self.parameters.get("sway_sampling_coef", "-1.0"))
        self.speed = float(self.parameters.get("speed", "1.0"))
        self.cross_fade_duration = float(self.parameters.get("cross_fade_duration", "0.15"))
        self.fix_duration = _decode_optional_float(self.parameters.get("fix_duration", ""))

        vocoder_name = self.parameters.get("vocoder_name", model_cfg.model.mel_spec.mel_spec_type)
        load_vocoder_from_local = _decode_bool(self.parameters.get("load_vocoder_from_local", "false"), default=False)
        vocoder_local_path = self.parameters.get("vocoder_local_path", "")
        self.vocoder_name = vocoder_name

        ckpt_file = self.parameters.get("ckpt_file", "")
        if not ckpt_file:
            if model_name == "F5TTS_v1_Base":
                ckpt_file = "hf://SWivid/F5-TTS/F5TTS_v1_Base/model_1250000.safetensors"
            elif model_name == "F5TTS_Base":
                ckpt_file = "hf://SWivid/F5-TTS/F5TTS_Base/model_1200000.safetensors"
            elif model_name == "E2TTS_Base":
                ckpt_file = "hf://SWivid/E2-TTS/E2TTS_Base/model_1200000.safetensors"
            else:
                raise ValueError(f"Unsupported model_name for default ckpt resolution: {model_name}")
        if ckpt_file.startswith("hf://"):
            ckpt_file = str(cached_path(ckpt_file))
        base_ckpt_sha256 = file_sha256(ckpt_file)

        vocab_file = self.parameters.get("vocab_file", "")
        if vocab_file.startswith("hf://"):
            vocab_file = str(cached_path(vocab_file))

        self.vocoder = load_vocoder(
            vocoder_name=vocoder_name,
            is_local=load_vocoder_from_local,
            local_path=vocoder_local_path,
            device=self.device,
        )
        self.model = load_model(
            model_cls,
            model_arc,
            ckpt_file,
            mel_spec_type=vocoder_name,
            vocab_file=vocab_file,
            device=self.device,
        )

        adapter_cfg = PVCAdapterConfig(
            rank=int(self.parameters.get("lora_rank", "16")),
            alpha=float(self.parameters.get("lora_alpha", "16")),
            lora_dropout=float(self.parameters.get("lora_dropout", "0.05")),
            prompt_drop_path=float(self.parameters.get("prompt_drop_path", "0.3")),
            conditioning_enabled=_decode_bool(self.parameters.get("conditioning_enabled", "true"), default=True),
            conditioning_gamma=float(self.parameters.get("conditioning_gamma", "0.25")),
            conditioning_kernel_size=int(self.parameters.get("conditioning_kernel_size", "3")),
            conditioning_se_reduction=int(self.parameters.get("conditioning_se_reduction", "4")),
            conditioning_target_regex=self.parameters.get(
                "conditioning_target_regex",
                PVCAdapterConfig().conditioning_target_regex,
            ),
        )
        apply_pvc_adapters(self.model, adapter_cfg)

        registry_file = self.parameters.get("adapter_registry", "")
        if registry_file and not os.path.isabs(registry_file):
            registry_file = os.path.abspath(os.path.join(model_dir, registry_file))
        adapter_registry = {}
        if registry_file and os.path.isfile(registry_file):
            with open(registry_file, "r", encoding="utf-8") as f:
                adapter_registry = json.load(f)

        lora_root = self.parameters.get("lora_root", "")
        if lora_root and not os.path.isabs(lora_root):
            lora_root = os.path.abspath(os.path.join(model_dir, lora_root))

        self.adapter_manager = AdapterManager(
            model=self.model,
            device=self.device,
            lora_root=lora_root,
            adapter_registry=adapter_registry,
            gpu_cache_size=int(self.parameters.get("adapter_cache_size_gpu", "8")),
            cpu_cache_size=int(self.parameters.get("adapter_cache_size_cpu", "64")),
            strict=_decode_bool(self.parameters.get("strict_adapter", "true"), default=True),
            expected_model_name=model_name,
            expected_base_ckpt_sha256=base_ckpt_sha256,
        )
        self.request_count = 0
        self.log_metrics_every_n_requests = int(self.parameters.get("log_metrics_every_n_requests", "100"))

        if self.reference_sample_rate != 24000:
            self.resampler = torchaudio.transforms.Resample(self.reference_sample_rate, 24000)
        else:
            self.resampler = None

    def _decode_string(self, request, name: str, default: str = ""):
        tensor = pb_utils.get_input_tensor_by_name(request, name)
        if tensor is None:
            return default
        value = tensor.as_numpy().reshape(-1)[0]
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="ignore")
        return str(value)

    def _decode_wave(self, request):
        wav_tensor = pb_utils.get_input_tensor_by_name(request, "reference_wav")
        if wav_tensor is None:
            raise pb_utils.TritonModelException("Missing required input: reference_wav")

        wav = from_dlpack(wav_tensor.to_dlpack()).float().cpu()
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)
        elif wav.ndim > 2:
            wav = wav.reshape(1, -1)

        wav_len_tensor = pb_utils.get_input_tensor_by_name(request, "reference_wav_len")
        if wav_len_tensor is not None:
            wav_len = int(from_dlpack(wav_len_tensor.to_dlpack()).reshape(-1)[0].item())
            wav = wav[:, :wav_len]

        if wav.shape[0] > 1:
            wav = torch.mean(wav, dim=0, keepdim=True)
        if wav.shape[-1] == 0:
            raise pb_utils.TritonModelException("reference_wav is empty.")

        return wav

    def _normalize_ref_text(self, ref_text: str):
        ref_text = ref_text.strip()
        if not ref_text:
            raise pb_utils.TritonModelException("reference_text cannot be empty for PVC Triton model.")
        if not ref_text.endswith(". ") and not ref_text.endswith("。"):
            if ref_text.endswith("."):
                ref_text += " "
            else:
                ref_text += ". "
        return ref_text

    def _run_single(self, waveform: torch.Tensor, ref_text: str, target_text: str):
        ref_text = self._normalize_ref_text(ref_text)
        if not target_text.strip():
            raise pb_utils.TritonModelException("target_text cannot be empty.")

        sr = self.reference_sample_rate
        if self.resampler is not None:
            waveform = self.resampler(waveform)
            sr = 24000
        audio = waveform

        duration_sec = max(0.01, audio.shape[-1] / sr)
        max_chars = int(len(ref_text.encode("utf-8")) / duration_sec * max(1.0, (22 - duration_sec)) * self.speed)
        max_chars = max(20, max_chars)
        batches = chunk_text(target_text, max_chars=max_chars)

        final_wave, _, _ = next(
            infer_batch_process(
                (audio, sr),
                ref_text,
                batches,
                self.model,
                self.vocoder,
                mel_spec_type=self.vocoder_name,
                progress=None,
                target_rms=self.target_rms,
                cross_fade_duration=self.cross_fade_duration,
                nfe_step=self.nfe_step,
                cfg_strength=self.cfg_strength,
                sway_sampling_coef=self.sway_sampling_coef,
                speed=self.speed,
                fix_duration=self.fix_duration,
                device=self.device,
            )
        )
        if final_wave is None:
            raise pb_utils.TritonModelException("Inference returned no audio.")
        return final_wave.astype(np.float32)

    def execute(self, requests):
        responses = [None] * len(requests)
        parsed = []
        for idx, request in enumerate(requests):
            try:
                parsed.append(
                    {
                        "idx": idx,
                        "request": request,
                        "waveform": self._decode_wave(request),
                        "ref_text": self._decode_string(request, "reference_text", ""),
                        "target_text": self._decode_string(request, "target_text", ""),
                        "speaker_id": self._decode_string(request, "speaker_id", ""),
                        "adapter_id": self._decode_string(request, "adapter_id", ""),
                        "adapter_revision": self._decode_string(request, "adapter_revision", ""),
                    }
                )
            except Exception as e:
                responses[idx] = pb_utils.InferenceResponse(error=pb_utils.TritonError(f"Bad request: {e}"))

        grouped = OrderedDict()
        for item in parsed:
            group_key = (item["speaker_id"], item["adapter_id"], item["adapter_revision"])
            grouped.setdefault(group_key, []).append(item)

        for group_key, items in grouped.items():
            speaker_id, adapter_id, adapter_revision = group_key
            try:
                self.adapter_manager.activate(
                    speaker_id=speaker_id,
                    adapter_id=adapter_id,
                    adapter_revision=adapter_revision,
                )
            except Exception as e:
                err = pb_utils.TritonError(f"Adapter activation failed for {group_key}: {e}")
                for item in items:
                    responses[item["idx"]] = pb_utils.InferenceResponse(error=err)
                continue

            for item in items:
                try:
                    audio = self._run_single(item["waveform"], item["ref_text"], item["target_text"])
                    output = pb_utils.Tensor("waveform", audio)
                    responses[item["idx"]] = pb_utils.InferenceResponse(output_tensors=[output])
                except Exception as e:
                    responses[item["idx"]] = pb_utils.InferenceResponse(
                        error=pb_utils.TritonError(f"Inference failed: {e}")
                    )

        for i, response in enumerate(responses):
            if response is None:
                responses[i] = pb_utils.InferenceResponse(error=pb_utils.TritonError("Request was not processed"))

        self.request_count += len(requests)
        if self.log_metrics_every_n_requests > 0 and self.request_count % self.log_metrics_every_n_requests == 0:
            print(f"[PVC Triton] metrics: {self.adapter_manager.get_metrics()}")

        return responses
