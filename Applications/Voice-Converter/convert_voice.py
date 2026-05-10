"""
Any-to-Any Voice Conversion using kNN-VC.

Replaces each WavLM feature frame of a source utterance with the
mean of its k=4 nearest neighbours from a target speaker's matching
set, then resynthesises via HiFi-GAN.

Usage:
    python convert_voice.py \
        --source source.wav \
        --target-dir ./target_speaker/ \
        --output converted_output.wav \
        --topk 4 \
        --loudness -20
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import torchaudio

REPO_PATH = str(Path(__file__).resolve().parents[2] / ".." / "knn-vc")
SAMPLE_RATE = 16000


def resolve_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_pipeline(device: str):
    print(f"[1/4] Loading kNN-VC pipeline  (device={device})")
    t0 = time.time()
    knnvc = torch.hub.load(
        REPO_PATH,
        "knn_vc",
        prematched=True,
        pretrained=True,
        device=device,
        source="local",
        trust_repo=True,
    )
    print(f"       Done in {time.time()-t0:.1f}s  "
          f"(WavLM-Large + HiFi-GAN prematched)")
    return knnvc


def collect_reference_paths(target_dir: Path) -> list[Path]:
    exts = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}
    paths = sorted(p for p in target_dir.iterdir() if p.suffix.lower() in exts)
    if not paths:
        sys.exit(f"No audio files found in {target_dir}")
    return paths


def extract_source_features(knnvc, source_path: Path):
    print(f"[2/4] Extracting source features  ({source_path.name})")
    t0 = time.time()
    query_seq = knnvc.get_features(str(source_path))
    print(f"       {query_seq.shape[0]} frames  ({time.time()-t0:.1f}s)")
    return query_seq


def build_matching_set(knnvc, ref_paths: list[Path]):
    print(f"[3/4] Building matching set  ({len(ref_paths)} reference file(s))")
    t0 = time.time()
    matching_set = knnvc.get_matching_set([str(p) for p in ref_paths])
    print(f"       {matching_set.shape[0]} frames pooled  ({time.time()-t0:.1f}s)")
    return matching_set


def convert_and_save(
    knnvc,
    query_seq: torch.Tensor,
    matching_set: torch.Tensor,
    output_path: Path,
    topk: int,
    loudness_db: float,
):
    print(f"[4/4] k-NN matching (k={topk}) → HiFi-GAN synthesis")
    t0 = time.time()
    out_wav = knnvc.match(
        query_seq,
        matching_set,
        topk=topk,
        tgt_loudness_db=loudness_db,
    )
    torchaudio.save(str(output_path), out_wav[None], SAMPLE_RATE)
    dur = out_wav.shape[0] / SAMPLE_RATE
    print(f"       {dur:.2f}s audio saved → {output_path}  ({time.time()-t0:.1f}s)")


def main():
    parser = argparse.ArgumentParser(
        description="kNN-VC: any-to-any voice conversion."
    )
    parser.add_argument("--source", type=Path, required=True,
                        help="Path to the source .wav file.")
    parser.add_argument("--target-dir", type=Path, required=True,
                        help="Directory of target speaker reference audios.")
    parser.add_argument("--output", type=Path, default=Path("converted_output.wav"),
                        help="Output .wav path.")
    parser.add_argument("--topk", type=int, default=4,
                        help="Number of nearest neighbours (default: 4).")
    parser.add_argument("--loudness", type=float, default=-20.0,
                        help="Target loudness in dB LUFS (default: -20).")
    args = parser.parse_args()

    if not args.source.exists():
        sys.exit(f"Source file not found: {args.source}")
    if not args.target_dir.is_dir():
        sys.exit(f"Target directory not found: {args.target_dir}")

    device = resolve_device()
    knnvc = load_pipeline(device)

    query_seq = extract_source_features(knnvc, args.source)
    ref_paths = collect_reference_paths(args.target_dir)
    matching_set = build_matching_set(knnvc, ref_paths)

    convert_and_save(
        knnvc, query_seq, matching_set,
        args.output, args.topk, args.loudness,
    )
    print("\nConversion complete.")


if __name__ == "__main__":
    main()
