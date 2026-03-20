import argparse
import os

import numpy as np
import requests
import soundfile as sf


def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--server-url", type=str, default="localhost:8000")
    parser.add_argument("--model-name", type=str, default="f5_tts_pvc")
    parser.add_argument(
        "--reference-audio",
        type=str,
        default="../../infer/examples/basic/basic_ref_en.wav",
    )
    parser.add_argument(
        "--reference-text",
        type=str,
        default="Some call me nature, others call me mother nature.",
    )
    parser.add_argument(
        "--target-text",
        type=str,
        default="I don't really care what you call me. I've been a silent spectator.",
    )
    parser.add_argument("--speaker-id", type=str, default="speaker_001")
    parser.add_argument("--adapter-id", type=str, default="")
    parser.add_argument("--adapter-revision", type=str, default="")
    parser.add_argument("--output-audio", type=str, default="tests/client_http_pvc.wav")
    return parser.parse_args()


def load_audio(wav_path):
    waveform, sample_rate = sf.read(wav_path)
    if sample_rate != 24000:
        from scipy.signal import resample

        waveform = resample(waveform, int(len(waveform) * (24000 / sample_rate)))
    return waveform.astype(np.float32)


def main():
    args = get_args()
    server_url = args.server_url
    if not server_url.startswith(("http://", "https://")):
        server_url = f"http://{server_url}"
    url = f"{server_url}/v2/models/{args.model_name}/infer"

    waveform = load_audio(args.reference_audio)
    lengths = np.array([[len(waveform)]], dtype=np.int32)
    waveform = waveform.reshape(1, -1)

    inputs = [
        {"name": "reference_wav", "shape": list(waveform.shape), "datatype": "FP32", "data": waveform.tolist()},
        {"name": "reference_wav_len", "shape": list(lengths.shape), "datatype": "INT32", "data": lengths.tolist()},
        {"name": "reference_text", "shape": [1, 1], "datatype": "BYTES", "data": [args.reference_text]},
        {"name": "target_text", "shape": [1, 1], "datatype": "BYTES", "data": [args.target_text]},
        {"name": "speaker_id", "shape": [1, 1], "datatype": "BYTES", "data": [args.speaker_id]},
    ]

    if args.adapter_id:
        inputs.append({"name": "adapter_id", "shape": [1, 1], "datatype": "BYTES", "data": [args.adapter_id]})
    if args.adapter_revision:
        inputs.append(
            {"name": "adapter_revision", "shape": [1, 1], "datatype": "BYTES", "data": [args.adapter_revision]}
        )

    payload = {"inputs": inputs}
    rsp = requests.post(url, headers={"Content-Type": "application/json"}, json=payload, verify=False)
    rsp.raise_for_status()
    result = rsp.json()
    audio = np.array(result["outputs"][0]["data"], dtype=np.float32)
    os.makedirs(os.path.dirname(args.output_audio), exist_ok=True)
    sf.write(args.output_audio, audio, 24000, "PCM_16")


if __name__ == "__main__":
    main()
