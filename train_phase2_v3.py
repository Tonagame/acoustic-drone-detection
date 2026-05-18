"""
train_phase2_v3.py
------------------
Phase 2v3  -  Multi-view hard-negative drone detection CNN.

CORE IDEA
  ONE CNN trained on ALL 5 spectral filter views simultaneously (filter
  augmentation).  Hard negatives make the model learn:
      drone + tank   -> drone       tank alone   -> no_drone
      drone + engine -> drone       engine alone -> no_drone
      drone + speech -> drone       speech alone -> no_drone

OUTPUT
  models/drone_cnn_phase2_v3_multiview_hardnegatives.pth

USAGE
  cd E:\\drone_detect
  python train_phase2_v3.py

  # quick pipeline test (500 examples/class, ~2 min):
  python train_phase2_v3.py --quick

  # full training:
  python train_phase2_v3.py
"""

import argparse
import random
import time
import sys
from pathlib import Path

import numpy as np
import scipy.signal as sig
import soundfile as sf
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio.transforms as T
import torchaudio.functional as FA
from torch.utils.data import Dataset, DataLoader

# -- Paths ------------------------------------------------------------------
ROOT       = Path(__file__).parent
MODELS_DIR = ROOT / "models"
DATA_DIR   = ROOT / "data"
DRONE_DIR  = DATA_DIR / "raw" / "drone"
NODRONE_DIR= DATA_DIR / "raw" / "no_drone"
NOISE_BASE = DATA_DIR / "noise"
RESULTS_DIR= ROOT / "results" / "phase2_v3"
CKPT_DIR   = RESULTS_DIR / "checkpoints"

for d in (MODELS_DIR, RESULTS_DIR, CKPT_DIR):
    d.mkdir(parents=True, exist_ok=True)

SAVE_PATH = MODELS_DIR / "drone_cnn_phase2_v3_multiview_hardnegatives.pth"

# -- Audio constants ---------------------------------------------------------
FS          = 16000
WIN_SAMPLES = 16000     # 1 second
HOP_SAMPLES = 8000      # 50% overlap
NOISE_FLOOR = 0.002

# -- Filter coefficients (build once at import time) -------------------------
_HP150 = sig.butter(4, 150,         btype='high', fs=FS, output='sos')
_HP250 = sig.butter(4, 250,         btype='high', fs=FS, output='sos')
_BP200 = sig.butter(4, [200, 6000], btype='band', fs=FS, output='sos')
_BP500 = sig.butter(4, [500, 6000], btype='band', fs=FS, output='sos')

VIEW_NAMES   = ['raw', 'HPF-150', 'HPF-250', 'BPF-200-6k', 'BPF-500-6k']
VIEW_WEIGHTS = np.array([0.05, 0.20, 0.25, 0.35, 0.15], dtype=np.float32)

# -- CPU mel-spectrogram (used inside Dataset workers) ----------------------
_mel_cpu = T.MelSpectrogram(
    sample_rate=FS, n_fft=512,
    win_length=400, hop_length=160,
    n_mels=64, power=2.0,
)


# ===========================================================================
#  Audio helpers
# ===========================================================================

def _norm_view(x: np.ndarray) -> np.ndarray:
    x = x - x.mean()
    pk = np.abs(x).max()
    return x / pk if pk > 1e-6 else x

def create_audio_views(x: np.ndarray) -> list:
    """Return list of 5 filtered float32 views, all peak-normalised."""
    x = x.astype(np.float64)
    x = x - x.mean()
    pk = np.abs(x).max()
    if pk < 1e-6:
        return [np.zeros_like(x, dtype=np.float32)] * 5
    x = x / pk
    return [
        x.astype(np.float32),
        _norm_view(sig.sosfilt(_HP150, x)).astype(np.float32),
        _norm_view(sig.sosfilt(_HP250, x)).astype(np.float32),
        _norm_view(sig.sosfilt(_BP200, x)).astype(np.float32),
        _norm_view(sig.sosfilt(_BP500, x)).astype(np.float32),
    ]

def mix_at_snr(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """Mix clean + noise at target SNR; peak-normalise output."""
    clean = clean.astype(np.float64)
    noise = noise.astype(np.float64)
    # Match lengths
    if len(noise) < len(clean):
        reps = int(np.ceil(len(clean) / len(noise)))
        noise = np.tile(noise, reps)
    if len(noise) > len(clean):
        start = random.randint(0, len(noise) - len(clean))
        noise = noise[start:start + len(clean)]
    p_clean = np.mean(clean ** 2) + 1e-12
    p_noise = np.mean(noise ** 2) + 1e-12
    scale   = np.sqrt(p_clean / (p_noise * 10 ** (snr_db / 10.0)))
    mixed   = clean + scale * noise
    pk      = np.abs(mixed).max()
    return (mixed / pk).astype(np.float32) if pk > 1e-6 else mixed.astype(np.float32)

def audio_to_logmel(wav: np.ndarray) -> torch.Tensor:
    """wav (float32 numpy) -> [64, T] log-mel tensor (CPU)."""
    pk = np.abs(wav).max()
    if pk < NOISE_FLOOR:
        return torch.zeros(64, 98)  # 98 frames ? 1 second
    w   = wav / pk
    t   = torch.from_numpy(w.astype(np.float32)).unsqueeze(0)
    mel = _mel_cpu(t)
    return torch.log10(mel + 1e-10).squeeze(0)  # [64, T]

def load_wav(path: Path) -> np.ndarray:
    """Read WAV -> mono float32 @ 16 kHz, peak-normalised."""
    audio, sr = sf.read(str(path), dtype='float32')
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != FS:
        t     = torch.from_numpy(audio).unsqueeze(0)
        audio = FA.resample(t, sr, FS).squeeze(0).numpy()
    pk = np.abs(audio).max()
    return audio / pk if pk > 1e-4 else audio

def window_audio(audio: np.ndarray, win=WIN_SAMPLES, hop=HOP_SAMPLES) -> list:
    """Slice audio into 1-second windows with 50% overlap."""
    wins = []
    for s in range(0, len(audio) - win + 1, hop):
        wins.append(audio[s:s + win].copy())
    return wins


# -- Synthetic noise fallbacks ----------------------------------------------
def _lp(s, taps):
    return np.convolve(s, np.ones(taps) / taps, mode='same')

def _norm(s, lv=0.85):
    p = np.abs(s).max()
    return (s / p * lv).astype(np.float32) if p > 1e-7 else s.astype(np.float32)

def synth_tank(n, t0=0.0):
    t   = np.linspace(t0, t0 + n / FS, n, endpoint=False)
    rpm = 1.0 + 0.04 * np.sin(2 * np.pi * 0.3 * t)
    f0  = 45.0
    eng = (0.55 * np.sin(2 * np.pi * f0 * rpm * t) +
           0.25 * np.sin(2 * np.pi * f0 * 2 * rpm * t) +
           0.12 * np.sin(2 * np.pi * f0 * 3 * rpm * t) +
           0.08 * np.sin(2 * np.pi * f0 * 4 * rpm * t))
    clank = np.zeros(n)
    rng   = np.random.default_rng(int(t0 * 100) % 9999)
    for pos in range(0, n, int(FS * 0.15)):
        b = min(int(FS * 0.01), n - pos)
        clank[pos:pos + b] = rng.standard_normal(b) * 0.4
    return _norm(eng + clank + _lp(np.random.randn(n), 64) * 0.3)

def synth_engine(n, t0=0.0):
    """
    Improved vehicle-engine synthesizer (v2).
    Random f0 (60-120 Hz) + heavy broadband exhaust noise (~same amplitude as
    harmonics) makes BPF-200-6k view look noise-dominated rather than harmonic,
    preventing the CNN from confusing engine with drone blade-pass lines.
    """
    rng = np.random.default_rng(int(t0 * 1000 + 17) % 99991)
    f0  = rng.uniform(60.0, 120.0)

    t   = np.linspace(t0, t0 + n / FS, n, endpoint=False)
    rpm = 1.0 + 0.05 * np.sin(2 * np.pi * 1.2 * t)

    # FM-style harmonics via instantaneous phase integration
    ph   = np.cumsum(rpm) * (f0 / FS) * 2 * np.pi
    harm = (0.55 * np.sin(ph) +
            0.25 * np.sin(2 * ph) +
            0.12 * np.sin(3 * ph) +
            0.06 * np.sin(4 * ph) +
            0.03 * np.sin(5 * ph))

    # Strong broadband exhaust (LP ~2 kHz) — similar amplitude to harmonics
    exhaust = _lp(rng.standard_normal(n), max(1, int(FS / 2000))) * 0.7

    # Irregular mechanical impulses (valve train, injectors)
    mech = np.zeros(n)
    pos  = 0
    while pos < n:
        pos += int(rng.integers(max(1, int(FS * 0.03)), max(2, int(FS * 0.12))))
        if pos >= n:
            break
        b = min(int(rng.integers(1, 6)), n - pos)
        if b > 0:
            mech[pos:pos + b] = rng.standard_normal(b) * rng.uniform(0.05, 0.3)

    return _norm(harm + exhaust + mech)

def synth_crowd(n, t0=0.0):
    white = np.random.randn(n)
    bp    = _lp(white - _lp(white, 80), 5)
    t     = np.linspace(t0, t0 + n / FS, n, endpoint=False)
    am    = 0.4 + 0.6 * np.abs(np.sin(2 * np.pi * 3.0 * t))
    return _norm(bp * am, 0.6)

SYNTH_FUNCS = {
    'tank':   synth_tank,
    'engine': synth_engine,
    'crowd':  synth_crowd,
}


# ===========================================================================
#  Data loading
# ===========================================================================

def collect_windows_from_files(folder: Path, max_wins: int) -> list:
    """Load WAV files from a folder -> list of WIN_SAMPLES numpy arrays."""
    wins = []
    if not folder.exists():
        return wins
    for f in sorted(folder.glob('*.wav')):
        if len(wins) >= max_wins:
            break
        try:
            audio = load_wav(f)
            wins.extend(window_audio(audio))
        except Exception:
            pass
    return wins[:max_wins]

def collect_noise_windows(noise_type: str, max_wins: int) -> list:
    """Load noise files; fall back to synthesis if folder is empty."""
    folder = NOISE_BASE / noise_type
    wins   = collect_windows_from_files(folder, max_wins)
    if not wins and noise_type in SYNTH_FUNCS:
        fn = SYNTH_FUNCS[noise_type]
        wins = [fn(WIN_SAMPLES, k * WIN_SAMPLES / FS) for k in range(max_wins)]
    return wins[:max_wins]

def split_files(files: list, frac_train=0.70, frac_val=0.15):
    """Split file list into (train, val, test) by source file."""
    n     = len(files)
    perm  = list(range(n))
    random.shuffle(perm)
    n_tr  = max(1, round(n * frac_train))
    n_va  = max(1, round(n * frac_val))
    return ([files[i] for i in perm[:n_tr]],
            [files[i] for i in perm[n_tr:n_tr + n_va]],
            [files[i] for i in perm[n_tr + n_va:]])


# ===========================================================================
#  Dataset
# ===========================================================================

class Phase2v3Dataset(Dataset):
    """
    Each __getitem__ call:
      DRONE  : pick a base drone window + random noise + random SNR
               -> mix -> random filter view -> log-mel -> label 0
      NODRONE: pick a noise-only window -> random filter view -> log-mel -> label 1

    Filter augmentation: the model sees all 5 views during training,
    one randomly per example per epoch.  This teaches the model to
    recognise drones under ANY of the 5 spectral projections.
    """

    def __init__(self, drone_wins, noise_wins, nodrone_wins,
                 n_drone, n_nodrone, snr_levels, augment=True):
        """
        drone_wins    : list of np.ndarray [WIN_SAMPLES]  (base drone audio)
        noise_wins    : dict  noise_type -> list of np.ndarray
        nodrone_wins  : list of np.ndarray  (noise-only, hard negatives)
        n_drone       : how many drone examples to expose per epoch
        n_nodrone     : how many no_drone examples to expose per epoch
        """
        self.drone_wins   = drone_wins
        self.noise_wins   = noise_wins
        self.nodrone_wins = nodrone_wins
        self.n_drone      = n_drone
        self.n_nodrone    = n_nodrone
        self.snr_levels   = snr_levels
        self.augment      = augment
        self.noise_types  = [k for k, v in noise_wins.items() if v]

    def __len__(self):
        return self.n_drone + self.n_nodrone

    def __getitem__(self, idx):
        if idx < self.n_drone:
            # -- DRONE example ----------------------------------------------
            dw    = self.drone_wins[idx % len(self.drone_wins)].copy()
            audio = dw

            if self.augment and self.noise_types:
                # Randomly decide whether to add noise (80% of the time)
                if random.random() < 0.80:
                    ntype = random.choice(self.noise_types)
                    nwins = self.noise_wins[ntype]
                    nwin  = nwins[random.randrange(len(nwins))]
                    snr   = random.choice(self.snr_levels)
                    audio = mix_at_snr(dw, nwin, snr)

                    # 20% chance: add a second noise type on top (harder)
                    if random.random() < 0.20 and len(self.noise_types) > 1:
                        ntype2 = random.choice(
                            [t for t in self.noise_types if t != ntype])
                        nwins2 = self.noise_wins[ntype2]
                        nwin2  = nwins2[random.randrange(len(nwins2))]
                        snr2   = random.choice(self.snr_levels)
                        audio  = mix_at_snr(audio, nwin2, snr2)

            label = 0  # drone

        else:
            # -- NO-DRONE example -------------------------------------------
            nd_idx = (idx - self.n_drone) % len(self.nodrone_wins)
            audio  = self.nodrone_wins[nd_idx].copy()
            label  = 1  # no_drone

        # -- Random filter view ---------------------------------------------
        views     = create_audio_views(audio)
        view_idx  = random.randrange(5)
        audio_view = views[view_idx]

        # -- Log-mel --------------------------------------------------------
        logmel = audio_to_logmel(audio_view)   # [64, T]

        return logmel.unsqueeze(0).float(), torch.tensor(label, dtype=torch.long)


# ===========================================================================
#  Model  (same DroneCNN as all previous phases)
# ===========================================================================

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
        return self.fc(self.gap(self.features(x)).view(x.size(0), -1))


# ===========================================================================
#  Training / evaluation helpers
# ===========================================================================

def run_epoch(model, loader, optimizer, criterion, device, train=True):
    model.train(train)
    total_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            if train:
                optimizer.zero_grad()
            out  = model(X)
            loss = criterion(out, y)
            if train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(y)
            correct    += (out.argmax(1) == y).sum().item()
            total      += len(y)
    return total_loss / total, correct / total


@torch.no_grad()
def predict_multiview(model, audio: np.ndarray, drone_idx: int, device) -> float:
    """Run all 5 views through the model; return weighted score."""
    views  = create_audio_views(audio)
    probs  = np.zeros(5, dtype=np.float32)
    for vi, view in enumerate(views):
        lm = audio_to_logmel(view)
        X  = lm.unsqueeze(0).unsqueeze(0).to(device)  # [1,1,64,T]
        sc = torch.softmax(model(X), dim=1)
        probs[vi] = sc[0, drone_idx].item()
    return float((VIEW_WEIGHTS * probs).sum())


# ===========================================================================
#  Post-training condition evaluation
# ===========================================================================

def run_condition_tests(model, drone_idx, device, n_chunks=600):
    """Same scenario structure as internal_test.py, using multiview scoring."""
    print("\n  +----------------------------------------------------------+")
    print(  "  |  Condition tests  (multiview weighted score, thr=0.50)  |")
    print(  "  +--------------------------------+----------+-------------+")
    print(  "  | Condition                      | MeanProb |  Rate@0.50  |")
    print(  "  +--------------------------------+----------+-------------+")

    # Build drone pool
    drone_pool = [
        f for f in sorted(DRONE_DIR.glob('*.wav'))
        if sf.info(str(f)).frames >= WIN_SAMPLES
    ]
    if not drone_pool:
        print("  |  [SKIP] no drone files found                           |")
        print("  +----------------------------------------------------------+")
        return

    _dq = []
    def next_drone_chunk():
        nonlocal _dq
        if not _dq:
            audio = load_wav(random.choice(drone_pool))
            for i in range(0, len(audio) - WIN_SAMPLES + 1, HOP_SAMPLES):
                _dq.append(audio[i:i + HOP_SAMPLES].copy())
        return _dq.pop(0)

    # Use stateful counter so synthetic noise advances t0 each chunk (avoids
    # identical periodic chunks that look drone-like to the CNN).
    def _make_synth(fn):
        ctr = [0]
        def gen():
            t0 = ctr[0] * HOP_SAMPLES / FS
            ctr[0] += 1
            return fn(HOP_SAMPLES, t0)
        return gen

    CONDITIONS = [
        # (label, gen_fn, expect_detect)
        ("drone alone",          lambda: next_drone_chunk(),                                                 True),
        ("drone + tank   0dB",   lambda: mix_at_snr(next_drone_chunk(), _make_synth(synth_tank)(),    0),   True),
        ("drone + tank  -5dB",   lambda: mix_at_snr(next_drone_chunk(), _make_synth(synth_tank)(),   -5),   True),
        ("drone + tank -10dB",   lambda: mix_at_snr(next_drone_chunk(), _make_synth(synth_tank)(),  -10),   True),
        ("drone + tank -20dB",   lambda: mix_at_snr(next_drone_chunk(), _make_synth(synth_tank)(),  -20),   True),
        ("tank alone",           _make_synth(synth_tank),                                                    False),
        ("engine alone",         _make_synth(synth_engine),                                                  False),
        ("crowd alone",          _make_synth(synth_crowd),                                                   False),
    ]

    model.eval()
    buf = np.zeros(WIN_SAMPLES, dtype=np.float32)
    results = {}
    for name, gen_fn, expect in CONDITIONS:
        _dq.clear()
        buf[:] = 0
        scores = []
        for i in range(n_chunks):
            chunk             = gen_fn()
            buf[:HOP_SAMPLES] = buf[HOP_SAMPLES:]
            buf[HOP_SAMPLES:] = chunk
            if i < 1:
                continue
            score = predict_multiview(model, buf.copy(), drone_idx, device)
            scores.append(score)
        mean_p = np.mean(scores)
        rate   = np.mean(np.array(scores) > 0.5) * 100
        tag    = "recall" if expect else "FA    "
        flag   = ("OK" if (rate > 50) == expect else "!!")
        print(f"  | {name:<30s}  |  {mean_p:.4f}  |  {tag} {rate:5.1f}%  {flag}  |")

    print("  +--------------------------------+----------+-------------+\n")


# ===========================================================================
#  Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick',      action='store_true',
                        help='Quick test: 500 examples/class, 5 epochs')
    parser.add_argument('--epochs',     type=int,   default=50)
    parser.add_argument('--batch',      type=int,   default=32)
    parser.add_argument('--lr',         type=float, default=1e-3)
    parser.add_argument('--max-per-class', type=int, default=10000,
                        help='Max training examples per class')
    parser.add_argument('--no-gpu',     action='store_true')
    parser.add_argument('--no-finetune',action='store_true',
                        help='Train from scratch even if a base model exists')
    parser.add_argument('--save-name',  type=str,   default=None,
                        help='Override output filename (basename only, placed in models/)')
    args = parser.parse_args()

    if args.quick:
        args.max_per_class = 500
        args.epochs        = 5
        print("  [quickTestMode] 500 examples/class, 5 epochs\n")

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    DEVICE = torch.device('cuda' if torch.cuda.is_available() and not args.no_gpu
                          else 'cpu')

    print("+==============================================================+")
    print("|   Phase 2v3  -  Multi-view Hard-Negative Drone CNN          |")
    print("+==============================================================+")
    print(f"  Device        : {DEVICE}")
    if DEVICE.type == 'cuda':
        props = torch.cuda.get_device_properties(0)
        print(f"  GPU           : {props.name}  ({props.total_memory/1e9:.1f} GB)")
    print(f"  Max/class     : {args.max_per_class}")
    # Allow caller to override the output filename
    global SAVE_PATH
    if args.save_name:
        SAVE_PATH = MODELS_DIR / args.save_name

    print(f"  Epochs        : {args.epochs}   Batch: {args.batch}   LR: {args.lr}")
    print(f"  Output        : {SAVE_PATH.name}\n")

    SNR_LEVELS = [-20, -15, -10, -5, 0, 5, 10]

    # -- Step 1: Collect source files ---------------------------------------
    print("-- Step 1: Collecting source files --------------------------")
    drone_files = sorted(DRONE_DIR.glob('*.wav')) if DRONE_DIR.exists() else []
    if not drone_files:
        sys.exit(f"  ERROR: No WAV files in {DRONE_DIR}")
    print(f"  Drone files : {len(drone_files)}")

    noise_types = ['tank', 'engine', 'wind', 'traffic', 'speech', 'crowd', 'custom']
    for nt in noise_types:
        nf = len(list((NOISE_BASE / nt).glob('*.wav'))) if (NOISE_BASE/nt).exists() else 0
        tag = f"{nf} files" if nf else "0 files (synthetic fallback)"
        print(f"  Noise {nt:<8}: {tag}")

    # File-level split (avoid data leakage across train/val/test)
    drone_tr, drone_va, drone_te = split_files(drone_files)
    print(f"\n  Drone split  train={len(drone_tr)}  val={len(drone_va)}  test={len(drone_te)}\n")

    # -- Step 2: Build audio window pools ----------------------------------
    print("-- Step 2: Loading audio windows ----------------------------")
    max_base = max(200, args.max_per_class // 5)   # base before augmentation

    def load_drone_split(files, limit):
        wins = []
        for f in files:
            if len(wins) >= limit: break
            try:
                wins.extend(window_audio(load_wav(f)))
            except Exception:
                pass
        random.shuffle(wins)
        return wins[:limit]

    drone_tr_wins = load_drone_split(drone_tr, max_base * 3)
    drone_va_wins = load_drone_split(drone_va, max_base)
    drone_te_wins = load_drone_split(drone_te, max_base)

    # Noise windows (shared across splits for synthesis; real files split too)
    noise_wins = {}
    for nt in noise_types:
        noise_wins[nt] = collect_noise_windows(nt, max_base)

    # No-drone windows = pure noise (all types), original no_drone clips
    nodrone_tr_wins = []
    for nt in noise_types:
        nodrone_tr_wins.extend(noise_wins[nt])
    # Original no_drone files
    if NODRONE_DIR.exists():
        nodrone_tr_wins.extend(collect_windows_from_files(NODRONE_DIR, max_base))
    random.shuffle(nodrone_tr_wins)
    nodrone_va_wins = nodrone_tr_wins[:max(50, len(nodrone_tr_wins)//5)]
    nodrone_te_wins = nodrone_tr_wins[len(nodrone_va_wins):len(nodrone_va_wins)*2]

    print(f"  Drone train windows  : {len(drone_tr_wins)}")
    print(f"  No-drone pool        : {len(nodrone_tr_wins)}")
    print(f"  Noise types loaded   : { {k:len(v) for k,v in noise_wins.items()} }\n")

    # -- Step 3: Create datasets --------------------------------------------
    print("-- Step 3: Building datasets --------------------------------")
    n_per_class_tr = min(args.max_per_class, len(drone_tr_wins) * 7)
    n_per_class_va = min(n_per_class_tr // 5, len(drone_va_wins) * 5)
    n_per_class_te = n_per_class_va

    ds_train = Phase2v3Dataset(
        drone_wins=drone_tr_wins, noise_wins=noise_wins,
        nodrone_wins=nodrone_tr_wins or [synth_tank(WIN_SAMPLES)],
        n_drone=n_per_class_tr, n_nodrone=n_per_class_tr,
        snr_levels=SNR_LEVELS, augment=True)

    ds_val = Phase2v3Dataset(
        drone_wins=drone_va_wins or drone_tr_wins[:50],
        noise_wins=noise_wins,
        nodrone_wins=nodrone_va_wins or nodrone_tr_wins[:50],
        n_drone=n_per_class_va, n_nodrone=n_per_class_va,
        snr_levels=SNR_LEVELS, augment=False)

    ds_test = Phase2v3Dataset(
        drone_wins=drone_te_wins or drone_tr_wins[:50],
        noise_wins=noise_wins,
        nodrone_wins=nodrone_te_wins or nodrone_tr_wins[50:100] if len(nodrone_tr_wins)>100 else nodrone_tr_wins[:50],
        n_drone=n_per_class_te, n_nodrone=n_per_class_te,
        snr_levels=SNR_LEVELS, augment=False)

    print(f"  Train : {len(ds_train)} samples  ({n_per_class_tr} drone + {n_per_class_tr} no_drone)")
    print(f"  Val   : {len(ds_val)} samples")
    print(f"  Test  : {len(ds_test)} samples")

    n_workers = 2 if DEVICE.type == 'cpu' else 4
    loader_tr = DataLoader(ds_train, batch_size=args.batch, shuffle=True,
                           num_workers=n_workers, pin_memory=(DEVICE.type=='cuda'))
    loader_va = DataLoader(ds_val,   batch_size=64, shuffle=False,
                           num_workers=n_workers, pin_memory=(DEVICE.type=='cuda'))
    loader_te = DataLoader(ds_test,  batch_size=64, shuffle=False,
                           num_workers=n_workers, pin_memory=(DEVICE.type=='cuda'))
    print()

    # -- Step 4: Build / load model ----------------------------------------
    print("-- Step 4: Preparing model ----------------------------------")
    model     = DroneCNN(n_classes=2).to(DEVICE)
    classes   = ['drone', 'no_drone']
    drone_idx = 0
    init_lr   = args.lr

    if not args.no_finetune:
        base_candidates = [
            MODELS_DIR / 'drone_cnn_phase3.pth',
            MODELS_DIR / 'drone_cnn_phase2v2.pth',
            MODELS_DIR / 'drone_cnn_phase2_noise_robust.pth',
            MODELS_DIR / 'drone_cnn_phase1b.pth',
            MODELS_DIR / 'drone_cnn_phase1.pth',
        ]
        for bc in base_candidates:
            if bc.exists():
                try:
                    ckpt = torch.load(str(bc), map_location=DEVICE, weights_only=False)
                    model.load_state_dict(ckpt['model_state_dict'], strict=False)
                    if 'classes' in ckpt:
                        cls = ckpt['classes']
                        if 'drone' in cls:
                            drone_idx = cls.index('drone')
                    init_lr = args.lr * 0.1   # 10? lower for fine-tuning
                    print(f"  Fine-tuning from  : {bc.name}  (LR -> {init_lr})")
                    break
                except Exception as e:
                    print(f"  Could not load {bc.name}: {e}")
        else:
            print("  No base model found -> training from scratch.")
    else:
        print("  Training from scratch (--no-finetune).")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params : {total_params:,}\n")

    # -- Step 5: Train -----------------------------------------------------
    print("-- Step 5: Training -----------------------------------------")
    optimizer = optim.Adam(model.parameters(), lr=init_lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.4, patience=3)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float('inf')
    patience_ctr  = 0
    PATIENCE      = 7

    print(f"  {'Epoch':>5}  {'TrainLoss':>10}  {'TrainAcc':>10}  "
          f"{'ValLoss':>10}  {'ValAcc':>10}  {'LR':>10}")
    print(f"  {'-'*65}")

    t_start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = run_epoch(model, loader_tr, optimizer, criterion, DEVICE, train=True)
        va_loss, va_acc = run_epoch(model, loader_va, optimizer, criterion, DEVICE, train=False)
        scheduler.step(va_loss)

        cur_lr = optimizer.param_groups[0]['lr']
        print(f"  {epoch:5d}  {tr_loss:10.4f}  {tr_acc*100:9.2f}%  "
              f"{va_loss:10.4f}  {va_acc*100:9.2f}%  {cur_lr:10.6f}")

        # Save best checkpoint
        if va_loss < best_val_loss:
            best_val_loss = va_loss
            patience_ctr  = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'classes':          classes,
                'epoch':            epoch,
                'val_loss':         va_loss,
                'val_acc':          va_acc,
                'phase':            'phase2v3',
                'description':      'multiview filter augmentation + hard negatives',
            }, str(CKPT_DIR / 'best_checkpoint.pth'))
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print(f"\n  Early stop at epoch {epoch} (no val improvement for {PATIENCE} epochs)")
                break

    elapsed = time.perf_counter() - t_start
    print(f"\n  Training done in {elapsed:.1f}s ({elapsed/60:.1f} min)\n")

    # Load best weights
    best = torch.load(str(CKPT_DIR / 'best_checkpoint.pth'),
                      map_location=DEVICE, weights_only=False)
    model.load_state_dict(best['model_state_dict'])
    print(f"  Best checkpoint: epoch {best['epoch']}  val_loss={best['val_loss']:.4f}  "
          f"val_acc={best['val_acc']*100:.2f}%\n")

    # -- Step 6: Test set evaluation ---------------------------------------
    print("-- Step 6: Test set evaluation ------------------------------")
    te_loss, te_acc = run_epoch(model, loader_te, optimizer, criterion, DEVICE, train=False)
    print(f"  Test loss : {te_loss:.4f}   Test acc : {te_acc*100:.2f}%\n")

    # -- Step 7: Per-condition evaluation ---------------------------------
    print("-- Step 7: Per-condition evaluation (multiview) -------------")
    model.eval()
    run_condition_tests(model, drone_idx, DEVICE, n_chunks=400)

    # -- Step 8: Threshold sweep -------------------------------------------
    print("-- Step 8: Threshold sweep ----------------------------------")
    THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7]
    model.eval()
    drone_pool = [f for f in sorted(DRONE_DIR.glob('*.wav'))
                  if sf.info(str(f)).frames >= WIN_SAMPLES]

    all_probs  = {'drone':[], 'tank':[], 'engine':[]}
    _dq2 = []
    def _next_drone():
        if not _dq2:
            audio = load_wav(random.choice(drone_pool))
            for i in range(0, len(audio)-WIN_SAMPLES+1, HOP_SAMPLES):
                _dq2.append(audio[i:i+HOP_SAMPLES].copy())
        return _dq2.pop(0)

    buf2 = np.zeros(WIN_SAMPLES, dtype=np.float32)
    for i in range(300):
        chunk=_next_drone(); buf2[:HOP_SAMPLES]=buf2[HOP_SAMPLES:]; buf2[HOP_SAMPLES:]=chunk
        if i>0: all_probs['drone'].append(predict_multiview(model, buf2.copy(), drone_idx, DEVICE))
    buf3 = np.zeros(WIN_SAMPLES, dtype=np.float32)
    for i in range(300):
        chunk=synth_tank(HOP_SAMPLES,i*HOP_SAMPLES/FS); buf3[:HOP_SAMPLES]=buf3[HOP_SAMPLES:]; buf3[HOP_SAMPLES:]=chunk
        if i>0: all_probs['tank'].append(predict_multiview(model, buf3.copy(), drone_idx, DEVICE))
    buf4 = np.zeros(WIN_SAMPLES, dtype=np.float32)
    for i in range(300):
        chunk=synth_engine(HOP_SAMPLES,i*HOP_SAMPLES/FS); buf4[:HOP_SAMPLES]=buf4[HOP_SAMPLES:]; buf4[HOP_SAMPLES:]=chunk
        if i>0: all_probs['engine'].append(predict_multiview(model, buf4.copy(), drone_idx, DEVICE))

    dp = np.array(all_probs['drone'])
    tp = np.array(all_probs['tank'])
    ep = np.array(all_probs['engine'])

    print(f"\n  {'Thr':>5}  {'DroneRecall':>12}  {'TankFA':>10}  {'EngineFA':>10}")
    print(f"  {'-'*45}")
    best_thr = 0.5
    for thr in THRESHOLDS:
        dr = np.mean(dp > thr) * 100
        tf = np.mean(tp > thr) * 100
        ef = np.mean(ep > thr) * 100
        print(f"  {thr:5.2f}  {dr:11.1f}%  {tf:9.1f}%  {ef:9.1f}%")
        if tf < 20 and dr > 50:
            best_thr = thr
    print(f"\n  Recommended threshold: {best_thr:.2f}\n")

    # -- Step 9: Save final model ------------------------------------------
    print("-- Step 9: Saving model -------------------------------------")
    torch.save({
        'model_state_dict': model.state_dict(),
        'classes':          classes,
        'phase':            'phase2v3',
        'description':      'multiview filter augmentation + hard negatives',
        'recommended_threshold': best_thr,
        'test_acc':         te_acc,
        'val_acc':          best['val_acc'],
        'snr_levels':       SNR_LEVELS,
        'view_names':       VIEW_NAMES,
        'view_weights':     VIEW_WEIGHTS.tolist(),
    }, str(SAVE_PATH))
    print(f"  Saved: {SAVE_PATH}\n")

    # -- Final summary -----------------------------------------------------
    print("+==============================================================+")
    print("|   Phase 2v3  DONE                                           |")
    print(f"|   Test accuracy       : {te_acc*100:5.2f}%                          |")
    print(f"|   Best val accuracy   : {best['val_acc']*100:5.2f}%                          |")
    print(f"|   Recommended thr     : {best_thr:.2f}                           |")
    print(f"|   Model               : {SAVE_PATH.name:<35s}|")
    print("+==============================================================+\n")
    print("  To test: update internal_test.py MODEL_PRIORITY to include")
    print(f"  'drone_cnn_phase2_v3_multiview_hardnegatives.pth'\n")


if __name__ == '__main__':
    main()
