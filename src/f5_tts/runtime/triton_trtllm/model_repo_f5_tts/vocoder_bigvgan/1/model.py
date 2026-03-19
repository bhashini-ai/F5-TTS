import json
import sys

import torch
import triton_python_backend_utils as pb_utils
from torch.utils.dlpack import from_dlpack, to_dlpack


class TritonPythonModel:
    def initialize(self, args):
        model_config = json.loads(args["model_config"])
        parameters = model_config.get("parameters", {})

        def _param(key, default=""):
            return parameters.get(key, {}).get("string_value", "").strip() or default

        model_dir = _param("model_dir", "/models/BigVGAN/bigvgan_v2_24khz_100band_256x")
        bigvgan_module_dir = _param("bigvgan_module_dir", "/workspace/F5-TTS/src/third_party/BigVGAN")

        self.device = torch.device("cuda")
        if bigvgan_module_dir not in sys.path:
            sys.path.append(bigvgan_module_dir)

        try:
            import bigvgan
        except Exception as exc:
            raise pb_utils.TritonModelException(
                f"Failed to import BigVGAN module from bigvgan_module_dir={bigvgan_module_dir}: {exc}"
            ) from exc

        try:
            self.vocoder = bigvgan.BigVGAN.from_pretrained(model_dir, use_cuda_kernel=False)
            self.vocoder.remove_weight_norm()
            self.vocoder = self.vocoder.eval().to(self.device)
        except Exception as exc:
            raise pb_utils.TritonModelException(
                f"Failed to load BigVGAN from model_dir={model_dir}: {exc}"
            ) from exc

    def execute(self, requests):
        responses = []
        for request in requests:
            try:
                mel_tensor = pb_utils.get_input_tensor_by_name(request, "mel")
                if mel_tensor is None:
                    raise ValueError("Missing required input tensor mel")

                mel = from_dlpack(mel_tensor.to_dlpack()).to(self.device, dtype=torch.float32)
                if mel.ndim == 2:
                    mel = mel.unsqueeze(0)
                if mel.ndim != 3:
                    raise ValueError(f"Expected mel shape [B,100,T], got {tuple(mel.shape)}")

                with torch.inference_mode():
                    waveform = self.vocoder(mel)

                if waveform.ndim == 3 and waveform.shape[1] == 1:
                    waveform = waveform.squeeze(1)
                if waveform.ndim == 2 and waveform.shape[0] == 1:
                    waveform = waveform.squeeze(0)

                waveform = waveform.to(torch.float32).contiguous().cpu().reshape(-1)
                output = pb_utils.Tensor.from_dlpack("waveform", to_dlpack(waveform))
                responses.append(pb_utils.InferenceResponse(output_tensors=[output]))
            except Exception as exc:
                responses.append(
                    pb_utils.InferenceResponse(
                        error=pb_utils.TritonError(f"vocoder_bigvgan failed: {exc}")
                    )
                )

        return responses

    def finalize(self):
        self.vocoder = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
