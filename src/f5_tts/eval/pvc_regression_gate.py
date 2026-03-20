from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from f5_tts.eval.utils_eval import run_asr_wer, run_sim


def parse_args():
    parser = argparse.ArgumentParser(description="PVC regression gate for adapter quality hardening")
    parser.add_argument("--manifest", type=str, required=True, help="JSONL with ref_wav/base_wav/pvc_wav/target_text")
    parser.add_argument("--lang", type=str, default="en", choices=["en", "zh"])
    parser.add_argument("--sim_ckpt", type=str, default="", help="Path to ECAPA checkpoint for speaker similarity")
    parser.add_argument("--asr_ckpt", type=str, default="", help="ASR checkpoint dir/model path if needed")
    parser.add_argument("--enable_utmos", action="store_true", help="Enable UTMOS scoring through torch.hub")
    parser.add_argument("--gpu_rank", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="tests/pvc_gate")

    parser.add_argument("--min_sim_delta", type=float, default=0.02, help="Required avg PVC-base similarity delta")
    parser.add_argument("--max_wer_delta", type=float, default=0.01, help="Allowed avg PVC-base WER delta")
    parser.add_argument("--min_utmos_delta", type=float, default=0.0, help="Required avg PVC-base UTMOS delta")
    parser.add_argument(
        "--require_metrics",
        action="store_true",
        help="If enabled, fail when any requested metric cannot be computed",
    )
    return parser.parse_args()


def load_manifest(path: str):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            for key in ["ref_wav", "base_wav", "pvc_wav", "target_text"]:
                if key not in payload:
                    raise ValueError(f"Missing '{key}' at line {line_no} in manifest")
            payload.setdefault("id", str(line_no))
            records.append(payload)
    if not records:
        raise ValueError("Manifest has no records")
    return records


def build_test_set(records, wav_key: str):
    return [(r[wav_key], r["ref_wav"], r["target_text"]) for r in records]


def by_stem(results):
    return {item["wav"]: item for item in results}


def compute_utmos_map(audio_paths, device):
    import librosa

    predictor = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True).to(device)
    scores = {}
    for p in audio_paths:
        wav, sr = librosa.load(p, sr=None, mono=True)
        wav_tensor = torch.from_numpy(wav).to(device).unsqueeze(0)
        scores[Path(p).stem] = predictor(wav_tensor, sr).item()
    return scores


def safe_mean(values):
    if not values:
        return None
    return sum(values) / len(values)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_manifest(args.manifest)
    base_test = build_test_set(records, "base_wav")
    pvc_test = build_test_set(records, "pvc_wav")

    metric_errors = []

    sim_base_map = {}
    sim_pvc_map = {}
    if args.sim_ckpt:
        try:
            sim_base_map = by_stem(run_sim((args.gpu_rank, base_test, args.sim_ckpt)))
            sim_pvc_map = by_stem(run_sim((args.gpu_rank, pvc_test, args.sim_ckpt)))
        except Exception as e:
            metric_errors.append(f"SIM failed: {e}")

    wer_base_map = {}
    wer_pvc_map = {}
    try:
        wer_base_map = by_stem(run_asr_wer((args.gpu_rank, args.lang, base_test, args.asr_ckpt)))
        wer_pvc_map = by_stem(run_asr_wer((args.gpu_rank, args.lang, pvc_test, args.asr_ckpt)))
    except Exception as e:
        metric_errors.append(f"WER failed: {e}")

    utmos_base_map = {}
    utmos_pvc_map = {}
    if args.enable_utmos:
        try:
            device = "cuda" if torch.cuda.is_available() else "xpu" if torch.xpu.is_available() else "cpu"
            utmos_base_map = compute_utmos_map([r["base_wav"] for r in records], device=device)
            utmos_pvc_map = compute_utmos_map([r["pvc_wav"] for r in records], device=device)
        except Exception as e:
            metric_errors.append(f"UTMOS failed: {e}")

    rows = []
    sim_deltas, wer_deltas, utmos_deltas = [], [], []
    for r in records:
        base_stem = Path(r["base_wav"]).stem
        pvc_stem = Path(r["pvc_wav"]).stem
        row = {"id": r["id"], "base_wav": r["base_wav"], "pvc_wav": r["pvc_wav"]}

        if base_stem in sim_base_map and pvc_stem in sim_pvc_map:
            row["sim_base"] = sim_base_map[base_stem]["sim"]
            row["sim_pvc"] = sim_pvc_map[pvc_stem]["sim"]
            row["sim_delta"] = row["sim_pvc"] - row["sim_base"]
            sim_deltas.append(row["sim_delta"])

        if base_stem in wer_base_map and pvc_stem in wer_pvc_map:
            row["wer_base"] = wer_base_map[base_stem]["wer"]
            row["wer_pvc"] = wer_pvc_map[pvc_stem]["wer"]
            row["wer_delta"] = row["wer_pvc"] - row["wer_base"]
            wer_deltas.append(row["wer_delta"])

        if base_stem in utmos_base_map and pvc_stem in utmos_pvc_map:
            row["utmos_base"] = utmos_base_map[base_stem]
            row["utmos_pvc"] = utmos_pvc_map[pvc_stem]
            row["utmos_delta"] = row["utmos_pvc"] - row["utmos_base"]
            utmos_deltas.append(row["utmos_delta"])

        rows.append(row)

    summary = {
        "num_records": len(records),
        "avg_sim_delta": safe_mean(sim_deltas),
        "avg_wer_delta": safe_mean(wer_deltas),
        "avg_utmos_delta": safe_mean(utmos_deltas),
        "metric_errors": metric_errors,
        "thresholds": {
            "min_sim_delta": args.min_sim_delta,
            "max_wer_delta": args.max_wer_delta,
            "min_utmos_delta": args.min_utmos_delta,
        },
    }

    checks = []
    if summary["avg_sim_delta"] is not None:
        checks.append(("sim_delta", summary["avg_sim_delta"] >= args.min_sim_delta, summary["avg_sim_delta"]))
    if summary["avg_wer_delta"] is not None:
        checks.append(("wer_delta", summary["avg_wer_delta"] <= args.max_wer_delta, summary["avg_wer_delta"]))
    if args.enable_utmos and summary["avg_utmos_delta"] is not None:
        checks.append(("utmos_delta", summary["avg_utmos_delta"] >= args.min_utmos_delta, summary["avg_utmos_delta"]))

    if args.require_metrics and metric_errors:
        checks.append(("metric_availability", False, "; ".join(metric_errors)))

    failed_checks = [c for c in checks if not c[1]]
    summary["checks"] = [{"name": n, "passed": p, "value": v} for n, p, v in checks]
    summary["gate_passed"] = len(failed_checks) == 0

    report = {"summary": summary, "rows": rows}

    json_path = output_dir / "pvc_regression_report.json"
    md_path = output_dir / "pvc_regression_report.md"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# PVC Regression Report\n\n")
        f.write(f"- Gate passed: `{summary['gate_passed']}`\n")
        f.write(f"- Records: `{summary['num_records']}`\n")
        f.write(f"- avg_sim_delta: `{summary['avg_sim_delta']}`\n")
        f.write(f"- avg_wer_delta: `{summary['avg_wer_delta']}`\n")
        f.write(f"- avg_utmos_delta: `{summary['avg_utmos_delta']}`\n")
        if metric_errors:
            f.write(f"- metric_errors: `{metric_errors}`\n")
        f.write("\n## Checks\n")
        for c in summary["checks"]:
            f.write(f"- {c['name']}: passed={c['passed']}, value={c['value']}\n")

    print(f"Wrote: {json_path}")
    print(f"Wrote: {md_path}")
    if not summary["gate_passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
