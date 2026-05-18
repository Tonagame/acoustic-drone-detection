"""
train_phase2_v4_specialist.py
-----------------------------
Phase 2v4 - five specialist CNNs, one fixed spectral view per model.

Each specialist sees only its own filtered projection during training. At
inference, the five drone probabilities are combined with the Phase 2v3
multiview weights and decision rule.

Usage:
  python train_phase2_v4_specialist.py
  python train_phase2_v4_specialist.py --quick
  python train_phase2_v4_specialist.py --epochs 30
  python train_phase2_v4_specialist.py --no-gpu
"""

import argparse
import copy
import random
import sys
import time
from pathlib import Path

import numpy as np
import scipy.signal as sig
import soundfile as sf
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio.functional as FA
import torchaudio.transforms as T
from torch.utils.data import DataLoader, Dataset

# -- Paths ------------------------------------------------------------------
ROOT        = Path(__file__).parent
MODELS_DIR  = ROOT / "models"
DATA_DIR    = ROOT / "data"
DRONE_DIR   = DATA_DIR / "raw" / "drone"
NODRONE_DIR = DATA_DIR / "raw" / "no_drone"
NOISE_BASE  = DATA_DIR / "noise"
RESULTS_DIR = ROOT / "results" / "phase2_v4"
CKPT_DIR    = RESULTS_DIR / "checkpoints"

for d in (MODELS_DIR, RESULTS_DIR, CKPT_DIR):
    d.mkdir(parents=True, exist_ok=True)

SAVE_PATH = MODELS_DIR / "drone_cnn_phase2_v4_specialist_ensemble.pth"

# -- Audio constants ---------------------------------------------------------
FS          = 16000
WIN_SAMPLES = 16000
HOP_SAMPLES = 8000
NOISE_FLOOR = 0.002

_HP150 = sig.butter(4, 150,         btype='high', fs=FS, output='sos')
_HP250 = sig.butter(4, 250,         btype='high', fs=FS, output='sos')
_BP200 = sig.butter(4, [200, 6000], btype='band', fs=FS, output='sos')
_BP500 = sig.butter(4, [500, 6000], btype='band', fs=FS, output='sos')

VIEW_NAMES   = ['raw', 'HPF-150', 'HPF-250', 'BPF-200-6k', 'BPF-500-6k']
VIEW_WEIGHTS = np.array([0.05, 0.20, 0.25, 0.35, 0.15], dtype=np.float32)

_mel_cpu = T.MelSpectrogram(
    sample_rate=FS, n_fft=512,
    win_length=400, hop_length=160,
    n_mels=64, power=2.0,
)


# ===========================================================================
#  Audio helpers copied from Phase 2v3
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
        return torch.zeros(64, 98)
    w   = wav / pk
    t   = torch.from_numpy(w.astype(np.float32)).unsqueeze(0)
    mel = _mel_cpu(t)
    return torch.log10(mel + 1e-10).squeeze(0)

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
    rng = np.random.default_rng(int(t0 * 1000 + 17) % 99991)
    f0  = rng.uniform(60.0, 120.0)
    t   = np.linspace(t0, t0 + n / FS, n, endpoint=False)
    rpm = 1.0 + 0.05 * np.sin(2 * np.pi * 1.2 * t)
    ph  = np.cumsum(rpm) * (f0 / FS) * 2 * np.pi
    harm = (0.55*np.sin(ph) + 0.25*np.sin(2*ph) + 0.12*np.sin(3*ph) +
            0.06*np.sin(4*ph) + 0.03*np.sin(5*ph))
    exhaust = _lp(rng.standard_normal(n), max(1, int(FS/2000))) * 0.7
    mech = np.zeros(n)
    pos = 0
    while pos < n:
        pos += int(rng.integers(max(1, int(FS*0.03)), max(2, int(FS*0.12))))
        if pos >= n: break
        b = min(int(rng.integers(1, 6)), n - pos)
        if b > 0:
            mech[pos:pos+b] = rng.standard_normal(b) * rng.uniform(0.05, 0.3)
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
#  Dataset and model
# ===========================================================================

class SpecialistDataset(Dataset):
    """
    Each sample uses ONLY the view at `view_idx` (0-4).
    No filter augmentation - this model specialises in exactly one projection.

    Label convention:
      0 = drone   (drone alone, drone+tank, drone+engine, drone+crowd, drone+speech)
      1 = no_drone (tank alone, engine alone, crowd alone, speech alone, pure noise)
    """
    def __init__(self, view_idx, drone_wins, noise_wins, nodrone_wins,
                 n_drone, n_nodrone, snr_levels, augment=True):
        self.view_idx     = view_idx
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
            dw    = self.drone_wins[idx % len(self.drone_wins)].copy()
            audio = dw
            if self.augment and self.noise_types:
                if random.random() < 0.80:
                    ntype = random.choice(self.noise_types)
                    nwin  = self.noise_wins[ntype][
                                random.randrange(len(self.noise_wins[ntype]))]
                    snr   = random.choice(self.snr_levels)
                    audio = mix_at_snr(dw, nwin, snr)
                    if random.random() < 0.20 and len(self.noise_types) > 1:
                        ntype2 = random.choice(
                            [t for t in self.noise_types if t != ntype])
                        nwin2  = self.noise_wins[ntype2][
                                     random.randrange(len(self.noise_wins[ntype2]))]
                        audio  = mix_at_snr(audio, nwin2, random.choice(self.snr_levels))
            label = 0
        else:
            nd_idx = (idx - self.n_drone) % len(self.nodrone_wins)
            audio  = self.nodrone_wins[nd_idx].copy()
            label  = 1

        views      = create_audio_views(audio)
        audio_view = views[self.view_idx]

        logmel = audio_to_logmel(audio_view)
        return logmel.unsqueeze(0).float(), torch.tensor(label, dtype=torch.long)


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


def run_epoch(model, loader, optimizer, criterion, device, train=True):
    model.train(train)
    total_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            if train:
                optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            if train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(y)
            correct    += (out.argmax(1) == y).sum().item()
            total      += len(y)
    return total_loss / max(total, 1), correct / max(total, 1)


# ===========================================================================
#  Ensemble inference and condition tests
# ===========================================================================

@torch.no_grad()
def predict_ensemble(models, audio, drone_idx, device):
    """Run 5 specialist models on their respective views. Return weighted score."""
    views = create_audio_views(audio)
    probs = np.zeros(5, dtype=np.float32)
    for vi, (model, view) in enumerate(zip(models, views)):
        lm = audio_to_logmel(view)
        X  = lm.unsqueeze(0).unsqueeze(0).to(device)
        sc = torch.softmax(model(X), dim=1)
        probs[vi] = sc[0, drone_idx].item()
    return float((VIEW_WEIGHTS * probs).sum()), probs

FMAX_THR  = 0.75
SCORE_THR = 0.60
VOTE_THR  = 0.60
VOTES_NEED = 2

def is_detection(probs):
    ws  = float(VIEW_WEIGHTS @ probs)
    fm  = float(probs[1:].max())
    vc  = int((probs > VOTE_THR).sum())
    return (fm > FMAX_THR) or (ws > SCORE_THR) or (vc >= VOTES_NEED), ws, fm, vc

def synth_pure_noise(n, t0=0.0):
    rng = np.random.default_rng(int(t0 * 1000 + 71) % 99991)
    return _norm(rng.standard_normal(n), 0.45)

def _condition_verdict(expect_detect, rate):
    if expect_detect:
        if rate >= 80.0:
            return "PASS"
        if rate >= 50.0:
            return "WARN"
        return "FAIL"
    if rate == 0.0:
        return "PASS"
    if rate <= 5.0:
        return "WARN"
    return "FAIL"

def build_drone_window_pool(max_files=1500):
    files = [f for f in sorted(DRONE_DIR.glob('*.wav'))
             if sf.info(str(f)).frames >= WIN_SAMPLES]
    if max_files and len(files) > max_files:
        files = random.sample(files, max_files)
    wins = []
    for f in files:
        try:
            wins.extend(window_audio(load_wav(f)))
        except Exception:
            pass
    random.shuffle(wins)
    return wins

def run_condition_tests(models, drone_idx, device, n_chunks=600):
    for model in models:
        model.eval()

    drone_wins = build_drone_window_pool()
    if not drone_wins:
        print("  [SKIP] condition tests: no >=1s drone files found")
        return

    drone_pos = [0]
    def next_drone_window():
        if drone_pos[0] >= len(drone_wins):
            random.shuffle(drone_wins)
            drone_pos[0] = 0
        w = drone_wins[drone_pos[0]]
        drone_pos[0] += 1
        return w.copy()

    scenarios = [
        ("drone alone",  True,  lambda i: next_drone_window()),
        ("drone+tank",   True,  lambda i: mix_at_snr(next_drone_window(), synth_tank(WIN_SAMPLES, i), 0)),
        ("drone+engine", True,  lambda i: mix_at_snr(next_drone_window(), synth_engine(WIN_SAMPLES, i), 0)),
        ("drone+crowd",  True,  lambda i: mix_at_snr(next_drone_window(), synth_crowd(WIN_SAMPLES, i), 0)),
        ("tank alone",   False, lambda i: synth_tank(WIN_SAMPLES, i)),
        ("engine alone", False, lambda i: synth_engine(WIN_SAMPLES, i)),
        ("crowd alone",  False, lambda i: synth_crowd(WIN_SAMPLES, i)),
        ("pure noise",   False, lambda i: synth_pure_noise(WIN_SAMPLES, i)),
    ]

    print("\n+================================================================================================+")
    print("| Phase 2v4 condition tests (600 one-second windows, specialist outputs + ensemble decision)      |")
    print("+----------------+----------+--------+--------+--------+--------+--------+----------+-----------+")
    print("| Scenario       | Det%     | raw    | HP150  | HP250  | BP200  | BP500  | Mean ws  | Verdict   |")
    print("+----------------+----------+--------+--------+--------+--------+--------+----------+-----------+")

    for name, expect, gen_fn in scenarios:
        dets = []
        weighted = []
        per_view = [[] for _ in range(5)]

        for i in range(n_chunks):
            audio = gen_fn(i)
            _, probs = predict_ensemble(models, audio, drone_idx, device)
            det, ws, _, _ = is_detection(probs)
            dets.append(det)
            weighted.append(ws)
            for vi in range(5):
                per_view[vi].append(probs[vi])

        rate = float(np.mean(dets) * 100.0)
        mean_ws = float(np.mean(weighted))
        view_rates = [float(np.mean(np.array(v) > VOTE_THR) * 100.0)
                      for v in per_view]
        verdict = _condition_verdict(expect, rate)
        print(f"| {name:<14s} | {rate:7.1f}% | "
              f"{view_rates[0]:5.1f}% | {view_rates[1]:5.1f}% | "
              f"{view_rates[2]:5.1f}% | {view_rates[3]:5.1f}% | "
              f"{view_rates[4]:5.1f}% | {mean_ws:8.3f} | {verdict:<9s} |")

    print("+----------------+----------+--------+--------+--------+--------+--------+----------+-----------+\n")


# ===========================================================================
#  Training driver
# ===========================================================================

def _safe_view_name(name):
    return name.replace("-", "_").replace("+", "_")

def make_datasets(view_idx, pools, args, snr_levels):
    return (
        SpecialistDataset(
            view_idx=view_idx,
            drone_wins=pools["drone_tr"],
            noise_wins=pools["noise"],
            nodrone_wins=pools["nodrone_tr"],
            n_drone=pools["n_tr"],
            n_nodrone=pools["n_tr"],
            snr_levels=snr_levels,
            augment=True),
        SpecialistDataset(
            view_idx=view_idx,
            drone_wins=pools["drone_va"],
            noise_wins=pools["noise"],
            nodrone_wins=pools["nodrone_va"],
            n_drone=pools["n_va"],
            n_nodrone=pools["n_va"],
            snr_levels=snr_levels,
            augment=False),
        SpecialistDataset(
            view_idx=view_idx,
            drone_wins=pools["drone_te"],
            noise_wins=pools["noise"],
            nodrone_wins=pools["nodrone_te"],
            n_drone=pools["n_te"],
            n_nodrone=pools["n_te"],
            snr_levels=snr_levels,
            augment=False),
    )

def train_one_specialist(view_idx, pools, args, device, snr_levels):
    view_name = VIEW_NAMES[view_idx]
    print(f"\n-- Specialist {view_idx}: {view_name} --------------------------------")

    ds_train, ds_val, ds_test = make_datasets(view_idx, pools, args, snr_levels)
    n_workers = 0 if args.no_workers else (2 if device.type == 'cpu' else 4)
    loader_tr = DataLoader(ds_train, batch_size=args.batch, shuffle=True,
                           num_workers=n_workers, pin_memory=(device.type == 'cuda'))
    loader_va = DataLoader(ds_val, batch_size=64, shuffle=False,
                           num_workers=n_workers, pin_memory=(device.type == 'cuda'))
    loader_te = DataLoader(ds_test, batch_size=64, shuffle=False,
                           num_workers=n_workers, pin_memory=(device.type == 'cuda'))

    model = DroneCNN(n_classes=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=4, min_lr=1e-5)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = -1.0
    best_val_loss = float('inf')
    best_epoch = 0
    best_state = None
    patience_ctr = 0
    ckpt_path = CKPT_DIR / f"best_{_safe_view_name(view_name)}.pth"

    print(f"  {'Epoch':>5}  {'TrainLoss':>10}  {'TrainAcc':>10}  "
          f"{'ValLoss':>10}  {'ValAcc':>10}  {'LR':>10}")
    print(f"  {'-' * 65}")

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = run_epoch(model, loader_tr, optimizer, criterion, device, train=True)
        va_loss, va_acc = run_epoch(model, loader_va, optimizer, criterion, device, train=False)
        scheduler.step(va_loss)
        cur_lr = optimizer.param_groups[0]['lr']

        print(f"  {epoch:5d}  {tr_loss:10.4f}  {tr_acc*100:9.2f}%  "
              f"{va_loss:10.4f}  {va_acc*100:9.2f}%  {cur_lr:10.6f}")

        improved = (va_acc > best_val_acc) or (
            va_acc == best_val_acc and va_loss < best_val_loss)
        if improved:
            best_val_acc = va_acc
            best_val_loss = va_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_ctr = 0
            torch.save({
                'model_state_dict': best_state,
                'classes': ['drone', 'no_drone'],
                'view_idx': view_idx,
                'view_name': view_name,
                'epoch': epoch,
                'val_loss': va_loss,
                'val_acc': va_acc,
                'phase': 'phase2v4',
            }, str(ckpt_path))
        else:
            patience_ctr += 1
            if patience_ctr >= 8:
                print(f"  Early stop at epoch {epoch} (no val_acc improvement for 8 epochs)")
                break

    if best_state is None:
        best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    te_loss, te_acc = run_epoch(model, loader_te, optimizer, criterion, device, train=False)
    print(f"  Best {view_name}: epoch={best_epoch} val_acc={best_val_acc*100:.2f}% "
          f"test_acc={te_acc*100:.2f}% -> {ckpt_path}")
    return model, best_state, {
        'view_idx': view_idx,
        'view_name': view_name,
        'best_epoch': best_epoch,
        'val_acc': best_val_acc,
        'val_loss': best_val_loss,
        'test_acc': te_acc,
        'test_loss': te_loss,
    }

def build_pools(args):
    print("-- Step 1: Collecting source files --------------------------")
    drone_files = sorted(DRONE_DIR.glob('*.wav')) if DRONE_DIR.exists() else []
    if not drone_files:
        sys.exit(f"  ERROR: No WAV files in {DRONE_DIR}")
    print(f"  Drone files : {len(drone_files)}")

    noise_types = ['tank', 'engine', 'wind', 'traffic', 'speech', 'crowd', 'custom']
    for nt in noise_types:
        nf = len(list((NOISE_BASE / nt).glob('*.wav'))) if (NOISE_BASE / nt).exists() else 0
        tag = f"{nf} files" if nf else "0 files (synthetic fallback if available)"
        print(f"  Noise {nt:<8}: {tag}")

    drone_tr, drone_va, drone_te = split_files(drone_files)
    print(f"\n  Drone split  train={len(drone_tr)}  val={len(drone_va)}  test={len(drone_te)}\n")

    print("-- Step 2: Loading audio windows ----------------------------")
    max_base = max(200, args.max_per_class // 5)

    def load_drone_split(files, limit):
        wins = []
        for f in files:
            if len(wins) >= limit:
                break
            try:
                wins.extend(window_audio(load_wav(f)))
            except Exception:
                pass
        random.shuffle(wins)
        return wins[:limit]

    drone_tr_wins = load_drone_split(drone_tr, max_base * 3)
    drone_va_wins = load_drone_split(drone_va, max_base)
    drone_te_wins = load_drone_split(drone_te, max_base)

    if not drone_tr_wins:
        sys.exit("  ERROR: No trainable drone windows found.")

    noise_wins = {nt: collect_noise_windows(nt, max_base) for nt in noise_types}
    noise_wins['pure_noise'] = [
        synth_pure_noise(WIN_SAMPLES, k * WIN_SAMPLES / FS)
        for k in range(max_base)
    ]

    nodrone_tr_wins = []
    for nt in list(noise_wins.keys()):
        nodrone_tr_wins.extend(noise_wins[nt])
    if NODRONE_DIR.exists():
        nodrone_tr_wins.extend(collect_windows_from_files(NODRONE_DIR, max_base))
    if not nodrone_tr_wins:
        nodrone_tr_wins = [synth_tank(WIN_SAMPLES), synth_engine(WIN_SAMPLES),
                           synth_crowd(WIN_SAMPLES), synth_pure_noise(WIN_SAMPLES)]
    random.shuffle(nodrone_tr_wins)
    nodrone_va_wins = nodrone_tr_wins[:max(50, len(nodrone_tr_wins) // 5)]
    nodrone_te_wins = nodrone_tr_wins[len(nodrone_va_wins):len(nodrone_va_wins) * 2]
    if not nodrone_te_wins:
        nodrone_te_wins = nodrone_va_wins or nodrone_tr_wins

    n_per_class_tr = min(args.max_per_class, len(drone_tr_wins) * 7)
    n_per_class_va = max(2, min(n_per_class_tr // 5, max(1, len(drone_va_wins)) * 5))
    n_per_class_te = n_per_class_va

    print(f"  Drone train windows  : {len(drone_tr_wins)}")
    print(f"  Drone val windows    : {len(drone_va_wins)}")
    print(f"  Drone test windows   : {len(drone_te_wins)}")
    print(f"  No-drone pool        : {len(nodrone_tr_wins)}")
    print(f"  Noise types loaded   : { {k: len(v) for k, v in noise_wins.items()} }")
    print(f"  Samples/model        : train={n_per_class_tr*2} val={n_per_class_va*2} test={n_per_class_te*2}\n")

    return {
        "drone_tr": drone_tr_wins,
        "drone_va": drone_va_wins or drone_tr_wins[:50],
        "drone_te": drone_te_wins or drone_tr_wins[:50],
        "noise": noise_wins,
        "nodrone_tr": nodrone_tr_wins,
        "nodrone_va": nodrone_va_wins or nodrone_tr_wins[:50],
        "nodrone_te": nodrone_te_wins or nodrone_tr_wins[:50],
        "n_tr": n_per_class_tr,
        "n_va": n_per_class_va,
        "n_te": n_per_class_te,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true',
                        help='Quick test: 500 examples/class, 5 epochs')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--max-per-class', type=int, default=10000)
    parser.add_argument('--no-gpu', action='store_true')
    parser.add_argument('--no-workers', action='store_true',
                        help='Use DataLoader num_workers=0 for debugging')
    parser.add_argument('--condition-chunks', type=int, default=600)
    args = parser.parse_args()

    if args.quick:
        args.max_per_class = 500
        args.epochs = 5
        print("  [quickTestMode] 500 examples/class, 5 epochs\n")

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_gpu else 'cpu')

    print("+==============================================================+")
    print("|   Phase 2v4 - 5-Specialist CNN Ensemble                     |")
    print("+==============================================================+")
    print(f"  Device        : {device}")
    if device.type == 'cuda':
        props = torch.cuda.get_device_properties(0)
        print(f"  GPU           : {props.name} ({props.total_memory/1e9:.1f} GB)")
    print(f"  Epochs/model  : {args.epochs}   Batch: {args.batch}   LR: {args.lr}")
    print(f"  Max/class     : {args.max_per_class}")
    print(f"  Output        : {SAVE_PATH.name}\n")

    snr_levels = [-20, -15, -10, -5, 0, 5, 10]
    pools = build_pools(args)

    models = []
    best_state_dicts = {}
    metrics = []
    start = time.perf_counter()
    for vi in range(5):
        model, best_state, info = train_one_specialist(vi, pools, args, device, snr_levels)
        models.append(model)
        best_state_dicts[vi] = best_state
        metrics.append(info)

    elapsed = time.perf_counter() - start
    print(f"\n-- Training complete in {elapsed:.1f}s ({elapsed/60:.1f} min) --------")

    bundle = {
        'phase':         'phase2v4',
        'hpf_hz':        0,
        'mel_fmin':      0.0,
        'view_names':    VIEW_NAMES,
        'view_weights':  VIEW_WEIGHTS.tolist(),
        'drone_idx':     0,
        'classes':       ['drone', 'no_drone'],
        'metrics':       metrics,
    }
    for vi, vname in enumerate(VIEW_NAMES):
        key = f'model_{vi}_{vname.replace("-","_").replace("+","_")}'
        bundle[key] = best_state_dicts[vi]
    torch.save(bundle, SAVE_PATH)
    print(f"Saved ensemble bundle -> {SAVE_PATH}")

    print("\n-- Post-training condition tests -----------------------------")
    run_condition_tests(models, drone_idx=0, device=device, n_chunks=args.condition_chunks)

    print("+==============================================================+")
    print("|   Phase 2v4 DONE                                            |")
    print(f"|   Model: {SAVE_PATH.name:<50s}|")
    print("+==============================================================+")


if __name__ == '__main__':
    main()
