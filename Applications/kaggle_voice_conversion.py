"""
k-NN Voice Conversion — Kaggle Notebook Script
Cells are separated by # %% markers for easy copy-paste.
Requires GPU runtime for reasonable speed.
"""

# %% [1] Dependencies
# !pip install torch torchaudio soundfile -q

# %% [2] Clone kNN-VC and install

# !git clone https://github.com/bshall/knn-vc.git
# !pip install numpy -q

# %% [3] Imports and Model Loading

import sys
from pathlib import Path
from urllib.request import urlretrieve

import torch
import torchaudio
import numpy as np

sys.path.insert(0, "knn-vc")
from hubconf import knn_vc

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAMPLE_RATE = 16000
OUTPUT_PATH = Path("converted_output.wav")

print(f"Device: {DEVICE}")
knnvc = knn_vc(pretrained=True, prematched=True, device=DEVICE)

# %% [4] Download Sample Audio

SAMPLES = {
    "source.wav": "https://github.com/bshall/knn-vc/releases/download/v0.1/david-attenborough.wav",
}

TARGET_DIR = Path("target_speaker")
TARGET_DIR.mkdir(exist_ok=True)

TARGET_REFS = {
    "ref_01.wav": "https://upload.wikimedia.org/wikipedia/commons/4/47/En-us-Hello.ogg",
}

for name, url in SAMPLES.items():
    if not Path(name).exists():
        print(f"Downloading {name} ...")
        urlretrieve(url, name)

for name, url in TARGET_REFS.items():
    dest = TARGET_DIR / name
    if not dest.exists():
        print(f"Downloading {name} ...")
        urlretrieve(url, str(dest))

print("Sample audio ready.")

# %% [5] Alternative: Record or Upload Your Own Audio
#
# For a proper demo, replace source.wav and target_speaker/ with real recordings.
# On Kaggle, you can upload files via the "Add Data" sidebar, or record locally
# and attach them as a dataset.
#
# Minimum requirements:
#   source.wav       — a few seconds of speech (the content you want to convert)
#   target_speaker/  — 1+ wav files of the target speaker (provides the voice timbre)

# %% [6] Feature Extraction

source_path = "source.wav"
ref_paths = sorted(str(p) for p in TARGET_DIR.iterdir()
                   if p.suffix.lower() in {".wav", ".flac", ".ogg", ".mp3"})

print(f"Source: {source_path}")
print(f"Target references: {len(ref_paths)} file(s)")

print("\nExtracting source features ...")
query_seq = knnvc.get_features(source_path)
print(f"  {query_seq.shape[0]} frames extracted")

print("Building target matching set ...")
matching_set = knnvc.get_matching_set(ref_paths)
print(f"  {matching_set.shape[0]} frames pooled")

# %% [7] k-NN Matching and Synthesis

TOPK = 4
LOUDNESS_DB = -20.0

print(f"\nRunning k-NN matching (k={TOPK}) and HiFi-GAN synthesis ...")
out_wav = knnvc.match(
    query_seq,
    matching_set,
    topk=TOPK,
    tgt_loudness_db=LOUDNESS_DB,
)

torchaudio.save(str(OUTPUT_PATH), out_wav[None], SAMPLE_RATE)
duration = out_wav.shape[0] / SAMPLE_RATE
print(f"Saved {duration:.2f}s of converted audio to {OUTPUT_PATH}")

# %% [8] Playback and Visualization

import matplotlib.pyplot as plt
from IPython.display import Audio, display

src_wav, src_sr = torchaudio.load(source_path)
if src_sr != SAMPLE_RATE:
    src_wav = torchaudio.functional.resample(src_wav, src_sr, SAMPLE_RATE)

fig, axes = plt.subplots(2, 1, figsize=(14, 5), sharex=False)

t_src = np.arange(src_wav.shape[1]) / SAMPLE_RATE
axes[0].plot(t_src, src_wav[0].numpy(), color="#1A237E", linewidth=0.4)
axes[0].set_title("Source Waveform", fontsize=12, fontweight="bold")
axes[0].set_ylabel("Amplitude")
axes[0].set_xlim(0, t_src[-1])

t_out = np.arange(out_wav.shape[0]) / SAMPLE_RATE
axes[1].plot(t_out, out_wav.numpy(), color="#F57C00", linewidth=0.4)
axes[1].set_title(f"Converted Waveform (k={TOPK})", fontsize=12, fontweight="bold")
axes[1].set_ylabel("Amplitude")
axes[1].set_xlabel("Time (s)")
axes[1].set_xlim(0, t_out[-1])

for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3)

plt.suptitle("kNN-VC: Any-to-Any Voice Conversion", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("voice_conversion_result.png", dpi=150, bbox_inches="tight")
plt.show()

print("\nSource audio:")
display(Audio(src_wav[0].numpy(), rate=SAMPLE_RATE))
print("\nConverted audio:")
display(Audio(out_wav.numpy(), rate=SAMPLE_RATE))
