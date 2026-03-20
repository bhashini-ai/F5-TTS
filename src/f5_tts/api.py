import random
import sys
import json
import os
from importlib.resources import files

import soundfile as sf
import tqdm
from cached_path import cached_path
from hydra.utils import get_class
from omegaconf import OmegaConf

from f5_tts.infer.utils_infer import (
    infer_process,
    load_model,
    load_vocoder,
    preprocess_ref_audio_text,
    remove_silence_for_generated_wav,
    save_spectrogram,
    transcribe,
)
from f5_tts.model.utils import seed_everything
from f5_tts.peft import (
    PVCAdapterConfig,
    apply_pvc_adapters,
    check_adapter_compatibility,
    clear_adapter,
    file_sha256,
    load_adapter as load_pvc_adapter,
)


class F5TTS:
    def __init__(
        self,
        model="F5TTS_v1_Base",
        ckpt_file="",
        vocab_file="",
        ode_method="euler",
        use_ema=True,
        vocoder_local_path=None,
        device=None,
        hf_cache_dir=None,
    ):
        self.model_name = model
        model_cfg = OmegaConf.load(str(files("f5_tts").joinpath(f"configs/{model}.yaml")))
        model_cls = get_class(f"f5_tts.model.{model_cfg.model.backbone}")
        model_arc = model_cfg.model.arch

        self.mel_spec_type = model_cfg.model.mel_spec.mel_spec_type
        self.target_sample_rate = model_cfg.model.mel_spec.target_sample_rate

        self.ode_method = ode_method
        self.use_ema = use_ema

        if device is not None:
            self.device = device
        else:
            import torch

            self.device = (
                "cuda"
                if torch.cuda.is_available()
                else "xpu"
                if torch.xpu.is_available()
                else "mps"
                if torch.backends.mps.is_available()
                else "cpu"
            )

        # Load models
        self.vocoder = load_vocoder(
            self.mel_spec_type, vocoder_local_path is not None, vocoder_local_path, self.device, hf_cache_dir
        )

        repo_name, ckpt_step, ckpt_type = "F5-TTS", 1250000, "safetensors"

        # override for previous models
        if model == "F5TTS_Base":
            if self.mel_spec_type == "vocos":
                ckpt_step = 1200000
            elif self.mel_spec_type == "bigvgan":
                model = "F5TTS_Base_bigvgan"
                ckpt_type = "pt"
        elif model == "E2TTS_Base":
            repo_name = "E2-TTS"
            ckpt_step = 1200000

        if not ckpt_file:
            ckpt_file = str(
                cached_path(f"hf://SWivid/{repo_name}/{model}/model_{ckpt_step}.{ckpt_type}", cache_dir=hf_cache_dir)
            )
        self.base_ckpt_file = ckpt_file
        self.base_ckpt_sha256 = file_sha256(ckpt_file)
        self.ema_model = load_model(
            model_cls, model_arc, ckpt_file, self.mel_spec_type, vocab_file, self.ode_method, self.use_ema, self.device
        )
        self.adapter_injected = False
        self.active_adapter_dir = None
        self.active_adapter_metadata = {}
        self.active_adapter_strict = True

    def transcribe(self, ref_audio, language=None):
        return transcribe(ref_audio, language)

    def export_wav(self, wav, file_wave, remove_silence=False):
        sf.write(file_wave, wav, self.target_sample_rate)

        if remove_silence:
            remove_silence_for_generated_wav(file_wave)

    def export_spectrogram(self, spec, file_spec):
        save_spectrogram(spec, file_spec)

    def _read_adapter_config(self, adapter_dir):
        cfg_file = os.path.join(adapter_dir, "adapter_config.json")
        if not os.path.exists(cfg_file):
            return {}

        with open(cfg_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def _build_adapter_cfg_from_payload(self, payload):
        default_cfg = PVCAdapterConfig()
        peft_cfg = payload.get("peft", {})
        return PVCAdapterConfig(
            rank=int(peft_cfg.get("rank", 16)),
            alpha=float(peft_cfg.get("alpha", 16.0)),
            lora_dropout=float(peft_cfg.get("lora_dropout", 0.05)),
            prompt_drop_path=float(peft_cfg.get("prompt_drop_path", 0.3)),
            prompt_target=peft_cfg.get("prompt_target", default_cfg.prompt_target),
            dit_target_regex=peft_cfg.get("dit_target_regex", default_cfg.dit_target_regex),
        )

    def _ensure_adapter_modules(self, adapter_cfg: PVCAdapterConfig):
        if self.adapter_injected:
            return
        apply_pvc_adapters(self.ema_model, adapter_cfg)
        self.adapter_injected = True

    def load_adapter(self, adapter_dir, strict=True):
        adapter_dir = os.path.abspath(adapter_dir)
        payload = self._read_adapter_config(adapter_dir)
        check_adapter_compatibility(
            payload,
            expected_model_name=self.model_name,
            expected_base_ckpt_sha256=self.base_ckpt_sha256,
            strict=strict,
        )
        adapter_cfg = self._build_adapter_cfg_from_payload(payload)
        self._ensure_adapter_modules(adapter_cfg)
        info = load_pvc_adapter(self.ema_model, adapter_dir, strict=strict)
        self.active_adapter_dir = adapter_dir
        self.active_adapter_metadata = info
        self.active_adapter_strict = strict
        return info

    def unload_adapter(self):
        if not self.adapter_injected:
            self.active_adapter_dir = None
            self.active_adapter_metadata = {}
            return 0
        cleared = clear_adapter(self.ema_model)
        self.active_adapter_dir = None
        self.active_adapter_metadata = {}
        return cleared

    def infer(
        self,
        ref_file,
        ref_text,
        gen_text,
        speaker_adapter=None,
        show_info=print,
        progress=tqdm,
        target_rms=0.1,
        cross_fade_duration=0.15,
        sway_sampling_coef=-1,
        cfg_strength=2,
        nfe_step=32,
        speed=1.0,
        fix_duration=None,
        remove_silence=False,
        file_wave=None,
        file_spec=None,
        seed=None,
    ):
        if seed is None:
            seed = random.randint(0, sys.maxsize)
        seed_everything(seed)
        self.seed = seed

        if speaker_adapter:
            speaker_adapter = os.path.abspath(speaker_adapter)
            if self.active_adapter_dir != speaker_adapter:
                self.load_adapter(speaker_adapter, strict=self.active_adapter_strict)

        ref_file, ref_text = preprocess_ref_audio_text(ref_file, ref_text, show_info=show_info)

        wav, sr, spec = infer_process(
            ref_file,
            ref_text,
            gen_text,
            self.ema_model,
            self.vocoder,
            self.mel_spec_type,
            show_info=show_info,
            progress=progress,
            target_rms=target_rms,
            cross_fade_duration=cross_fade_duration,
            nfe_step=nfe_step,
            cfg_strength=cfg_strength,
            sway_sampling_coef=sway_sampling_coef,
            speed=speed,
            fix_duration=fix_duration,
            device=self.device,
        )

        if file_wave is not None:
            self.export_wav(wav, file_wave, remove_silence)

        if file_spec is not None:
            self.export_spectrogram(spec, file_spec)

        return wav, sr, spec


if __name__ == "__main__":
    f5tts = F5TTS()

    wav, sr, spec = f5tts.infer(
        ref_file=str(files("f5_tts").joinpath("infer/examples/basic/basic_ref_en.wav")),
        ref_text="Some call me nature, others call me mother nature.",
        gen_text="I don't really care what you call me. I've been a silent spectator, watching species evolve, empires rise and fall. But always remember, I am mighty and enduring.",
        file_wave=str(files("f5_tts").joinpath("../../tests/api_out.wav")),
        file_spec=str(files("f5_tts").joinpath("../../tests/api_out.png")),
        seed=None,
    )

    print("seed :", f5tts.seed)
