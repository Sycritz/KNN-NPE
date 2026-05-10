"""
Quick setup for testing kNN-VC:
  1. Records your voice from the microphone (source)
  2. Downloads LibriSpeech samples as the target speaker reference
"""

import argparse
import os
import sys
import wave
from pathlib import Path
from urllib.request import urlretrieve

SAMPLE_RATE = 16000
CHANNELS = 1
DURATION = 5

BASE_DIR = Path(__file__).parent
SOURCE_PATH = BASE_DIR / "source.wav"
TARGET_DIR = BASE_DIR / "target_speaker"

LIBRISPEECH_SAMPLES = [
    "https://www.sample-videos.com/audio/mp3/crowd-cheering.mp3",
]

LIBRIVOX_SAMPLES = {
    "ref_01.wav": "https://github.com/bshall/knn-vc/releases/download/v0.1/-attenbdavidorough.wav",
}


def record_source(duration: int = DURATION):
    try:
        import pyaudio
    except ImportError:
        sys.exit("Install pyaudio: conda install -n torch pyaudio")

    print(f"\n🎤  Recording {duration}s of audio — speak now!")
    pa = pyaudio.PyAudio()
    stream = pa.open(
        format=pyaudio.paInt16,
        channels=CHANNELS,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=1024,
    )

    frames = []
    for _ in range(0, int(SAMPLE_RATE / 1024 * duration)):
        frames.append(stream.read(1024))

    stream.stop_stream()
    stream.close()
    pa.terminate()

    with wave.open(str(SOURCE_PATH), "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(b"".join(frames))

    print(f"   Saved → {SOURCE_PATH}")


def download_targets():
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n📥  Downloading target speaker reference(s) to {TARGET_DIR}/")

    for filename, url in LIBRIVOX_SAMPLES.items():
        dest = TARGET_DIR / filename
        if dest.exists():
            print(f"   {filename} — already exists, skipping")
            continue
        print(f"   Downloading {filename} ...")
        urlretrieve(url, str(dest))
        print(f"   Saved → {dest}")

    print(f"\n   {len(list(TARGET_DIR.glob('*.wav')))} reference file(s) ready.")


def main():
    parser = argparse.ArgumentParser(description="Setup test data for kNN-VC.")
    parser.add_argument("--skip-record", action="store_true",
                        help="Skip mic recording (use existing source.wav).")
    parser.add_argument("--duration", type=int, default=DURATION,
                        help="Recording duration in seconds (default: 5).")
    args = parser.parse_args()

    download_targets()

    if args.skip_record:
        if SOURCE_PATH.exists():
            print(f"\n✅  Using existing source: {SOURCE_PATH}")
        else:
            print(f"\n⚠️   No source.wav found. Run without --skip-record to record.")
    else:
        record_source(args.duration)

    print("\n" + "=" * 60)
    print("Ready! Run the conversion with:")
    print(f"\n  conda run -n torch python convert_voice.py \\")
    print(f"      --source {SOURCE_PATH} \\")
    print(f"      --target-dir {TARGET_DIR}/ \\")
    print(f"      --output converted_output.wav")
    print("=" * 60)


if __name__ == "__main__":
    main()
