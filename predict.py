"""
predict.py  --  Run the Phase 1 drone detector on any WAV file.

Usage
-----
    python predict.py  path/to/audio.wav
    python predict.py  path/to/audio.wav  --threshold 0.5

Output
------
Per-window prediction + an overall file-level verdict.

Exit codes
----------
    0  drone detected
    1  no drone detected
    2  error
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torchaudio.transforms as T

# ── Model definition (must match train_phase1_gpu.py) ────────────────────
class DroneCNN(nn.Module):
    def __init__(self, n_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc  = nn.Linear(64, n_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


# ── Config ────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent
# Auto-select best available model: phase2v2 > phase1b > phase1
def _best_model():
    for name in ("drone_cnn_phase3.pth",
                 "drone_cnn_phase2v2.pth",
                 "drone_cnn_phase1b.pth",
                 "drone_cnn_phase1.pth"):
        p = ROOT / "models" / name
        if p.exists():
            return p
    return ROOT / "models" / "drone_cnn_phase1.pth"
MODEL_PATH = _best_model()

FS          = 16000
WIN_SAMPLES = FS
HOP_SAMPLES = FS // 2

N_FFT   = 512
WIN_LEN = round(0.025 * FS)   # 400
HOP_LEN = WIN_LEN - round(0.015 * FS)  # 160
N_MELS  = 64

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_mel = T.MelSpectrogram(
    sample_rate=FS, n_fft=N_FFT,
    win_length=WIN_LEN, hop_length=HOP_LEN,
    n_mels=N_MELS, power=2.0,
)


# ── Helpers ───────────────────────────────────────────────────────────────
def load_model():
    ckpt    = torch.load(str(MODEL_PATH), map_location=DEVICE)
    classes = ckpt.get("classes", ["drone", "no_drone"])
    model   = DroneCNN(n_classes=len(classes)).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, classes


def preprocess(wav_path: Path):
    """Load WAV, mono, resample, normalise, slice into windows."""
    audio, sr = sf.read(str(wav_path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    if sr != FS:
        import torchaudio.functional as F
        t     = torch.from_numpy(audio).unsqueeze(0)
        audio = F.resample(t, sr, FS).squeeze(0).numpy()

    peak = np.abs(audio).max()
    if peak > 0:
        audio = audio / peak

    windows = []
    for s in range(0, len(audio) - WIN_SAMPLES + 1, HOP_SAMPLES):
        windows.append(audio[s : s + WIN_SAMPLES])

    return windows


def windows_to_features(windows):
    feats = []
    for w in windows:
        t      = torch.from_numpy(w).unsqueeze(0)          # [1, 16000]
        mel    = _mel(t)                                    # [1, 64, T]
        log_mel = torch.log10(mel + 1e-10).unsqueeze(0)    # [1, 1, 64, T]
        feats.append(log_mel)
    return torch.cat(feats, dim=0)   # [N, 1, 64, T]


# ── Main ──────────────────────────────────────────────────────────────────
def predict(wav_path: Path, threshold: float = 0.5, verbose: bool = True):
    if not wav_path.exists():
        print(f"ERROR: file not found: {wav_path}")
        sys.exit(2)

    if not MODEL_PATH.exists():
        print(f"ERROR: model not found at {MODEL_PATH}")
        print("Run train_phase1_gpu.py first.")
        sys.exit(2)

    model, classes = load_model()
    drone_idx = classes.index("drone")

    windows = preprocess(wav_path)
    if not windows:
        print(f"WARNING: audio shorter than 1 second — no windows to classify.")
        sys.exit(2)

    X = windows_to_features(windows).to(DEVICE)

    with torch.no_grad():
        logits = model(X)
        probs  = torch.softmax(logits, dim=1).cpu().numpy()

    drone_probs = probs[:, drone_idx]

    if verbose:
        print(f"\nFile   : {wav_path.name}")
        print(f"Device : {DEVICE}")
        print(f"Windows: {len(windows)}\n")
        print(f"{'Win':>4}  {'Drone prob':>10}  Verdict")
        print("-" * 30)
        for i, p in enumerate(drone_probs):
            verdict = "DRONE" if p >= threshold else "no drone"
            print(f"  {i+1:2d}   {p:9.4f}   {verdict}")

    # File-level decision: drone if ANY window exceeds threshold
    any_drone  = bool((drone_probs >= threshold).any())
    mean_prob  = float(drone_probs.mean())
    max_prob   = float(drone_probs.max())

    print(f"\n{'='*35}")
    print(f"  Mean drone prob : {mean_prob:.4f}")
    print(f"  Max  drone prob : {max_prob:.4f}")
    print(f"  Threshold       : {threshold}")
    print(f"  VERDICT         : {'*** DRONE DETECTED ***' if any_drone else 'no drone'}")
    print(f"{'='*35}\n")

    return any_drone, drone_probs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 1 drone audio detector")
    parser.add_argument("wav",       type=Path, help="Path to WAV file")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Drone probability threshold (default 0.5)")
    parser.add_argument("--quiet",   action="store_true",
                        help="Only print the final verdict")
    args = parser.parse_args()

    detected, _ = predict(args.wav, threshold=args.threshold,
                          verbose=not args.quiet)
    sys.exit(0 if detected else 1)
