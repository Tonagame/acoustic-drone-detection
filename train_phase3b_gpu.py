"""
train_phase3b_gpu.py  --  Phase 3b: HPF + Noise Robust Drone Detector

Key improvement over Phase 3:
  HIGH-PASS FILTER (cutoff 150 Hz) applied BEFORE mel spectrogram.
  This removes tank 45 Hz / engine 60-100 Hz harmonics entirely,
  while preserving drone propeller signal (200-2000 Hz).

  Result:  drone+tank 0dB mel  ≈  clean drone mel  (tank nearly invisible)
           tank-only     mel  ≈  near-silence      (correct: no drone)

Also:
  - Loads real no_drone WAV files (not just synthetic noise)
  - All noise synthesis on GPU — no CPU bottleneck
  - Trains from scratch (different feature space → can't reuse Phase 3 weights)
  - Preprocessing params saved in checkpoint so live_detector auto-adapts

Run:
    python train_phase3b_gpu.py
"""

import gc
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio.transforms as T
import torchaudio.functional as FA

ROOT       = Path(__file__).parent
MODELS_DIR = ROOT / "models"
DATA_DIR   = ROOT / "data" / "raw"
MODELS_DIR.mkdir(exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────
FS          = 16000
WIN_SAMPLES = 16000
HOP_SAMPLES = 8000

N_FFT   = 512
WIN_LEN = 400
HOP_LEN = 160
N_MELS  = 64

# ── KEY SIGNAL PROCESSING CHANGE ─────────────────────────────────────────
HPF_CUTOFF_HZ = 150   # Remove everything below 150 Hz before mel
F_MIN_MEL     = 150.0  # Mel filterbank also starts at 150 Hz
F_MAX_MEL     = 8000.0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ── Hyperparameters ───────────────────────────────────────────────────────
N_EPOCHS     = 35
BATCH_SIZE   = 128      # 64 drone + 64 no_drone
LR_START     = 3e-4     # from scratch
LR_MIN       = 1e-6
WEIGHT_DECAY = 1e-4

P_CLEAN        = 0.20
P_SINGLE_NOISE = 0.50
P_MULTI_NOISE  = 0.30

SNR_MIN = -10.0
SNR_MAX = +10.0

_NOISE_WEIGHTS = {
    "tank":       0.25,
    "engine":     0.20,
    "crowd":      0.10,
    "wind_light": 0.20,
    "wind_heavy": 0.20,
    "pink":       0.05,
}

# ── Mel transform with HPF-aware fmin ────────────────────────────────────
_mel_tf = T.MelSpectrogram(
    sample_rate=FS, n_fft=N_FFT,
    win_length=WIN_LEN, hop_length=HOP_LEN,
    n_mels=N_MELS, power=2.0,
    f_min=F_MIN_MEL,    # ← ignores all mel bins below 150 Hz
    f_max=F_MAX_MEL,
).to(DEVICE)


# ── High-Pass Filter (GPU, FFT-based, fully vectorized) ───────────────────
_hpf_mask = None    # built once, reused

def _get_hpf_mask(n: int) -> torch.Tensor:
    """Build (and cache) a binary mask for rfft output at length n."""
    global _hpf_mask
    if _hpf_mask is None or _hpf_mask.shape[0] != n // 2 + 1:
        freqs      = torch.fft.rfftfreq(n, d=1.0 / FS).to(DEVICE)
        _hpf_mask  = (freqs >= HPF_CUTOFF_HZ).float()
    return _hpf_mask

def hpf_gpu(wav: torch.Tensor) -> torch.Tensor:
    """
    Zero out FFT bins below HPF_CUTOFF_HZ.
    wav: [B, N] float32 on GPU  →  [B, N] filtered.
    """
    N    = wav.shape[-1]
    mask = _get_hpf_mask(N)
    X    = torch.fft.rfft(wav, n=N)
    X    = X * mask.unsqueeze(0)
    return torch.fft.irfft(X, n=N)


def audio_to_logmel(wav: torch.Tensor) -> torch.Tensor:
    """wav: [B, WIN_SAMPLES] → [B, 1, N_MELS, T] log-mel (HPF applied)."""
    wav_filt = hpf_gpu(wav)                          # remove <150 Hz
    mel      = _mel_tf(wav_filt)                     # [B, 64, T]
    lm       = torch.log10(mel + 1e-10)              # [B, 64, T]
    return lm.unsqueeze(1)                           # [B, 1, 64, T]


# ── Model (same DroneCNN — backward compatible) ───────────────────────────
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


# ── GPU Noise Synthesizers ────────────────────────────────────────────────

def _norm_batch(x: torch.Tensor, level: float = 0.85) -> torch.Tensor:
    peak = x.abs().amax(dim=1, keepdim=True).clamp(min=1e-7)
    return x / peak * level

def _fft_color(B: int, N: int, exponent: float) -> torch.Tensor:
    white = torch.randn(B, N, device=DEVICE)
    X     = torch.fft.rfft(white)
    freqs = torch.fft.rfftfreq(N, device=DEVICE).clamp(min=1e-6)
    X     = X / freqs.pow(exponent * 0.5).unsqueeze(0)
    sig   = torch.fft.irfft(X, n=N)
    return sig - sig.mean(dim=1, keepdim=True)

def synth_tank_gpu(B: int, N: int) -> torch.Tensor:
    """45 Hz diesel + harmonics + rumble (will be mostly HPF'd out)."""
    t   = torch.linspace(0, N / FS, N, device=DEVICE)
    rpm = 1.0 + 0.04 * torch.sin(2 * math.pi * 0.3 * t)
    f0  = 45.0
    eng = (0.55 * torch.sin(2 * math.pi * f0 * 1 * rpm * t) +
           0.25 * torch.sin(2 * math.pi * f0 * 2 * rpm * t) +
           0.12 * torch.sin(2 * math.pi * f0 * 3 * rpm * t) +
           0.08 * torch.sin(2 * math.pi * f0 * 4 * rpm * t)).expand(B, -1)
    rumble = _fft_color(B, N, 1.0) * 0.3
    noise  = torch.randn(B, N, device=DEVICE) * 0.08
    return _norm_batch(eng + rumble + noise)

def synth_engine_gpu(B: int, N: int) -> torch.Tensor:
    """60-100 Hz vehicle engine (randomised per sample)."""
    t   = torch.linspace(0, N / FS, N, device=DEVICE)
    rpm = 1.0 + 0.03 * torch.sin(2 * math.pi * 1.5 * t)
    f0  = torch.randint(60, 100, (B, 1), device=DEVICE).float()
    t_  = t.unsqueeze(0)
    s   = (0.50 * torch.sin(2 * math.pi * f0 * 1 * rpm * t_) +
           0.30 * torch.sin(2 * math.pi * f0 * 2 * rpm * t_) +
           0.12 * torch.sin(2 * math.pi * f0 * 3 * rpm * t_) +
           0.08 * torch.sin(2 * math.pi * f0 * 4 * rpm * t_))
    return _norm_batch(s + torch.randn(B, N, device=DEVICE) * 0.1)

def synth_crowd_gpu(B: int, N: int) -> torch.Tensor:
    """White noise with speech-rhythm AM (200+ Hz content survives HPF)."""
    white = torch.randn(B, N, device=DEVICE)
    t  = torch.linspace(0, N / FS, N, device=DEVICE).unsqueeze(0)
    f1 = torch.rand(B, 1, device=DEVICE) * 4 + 3
    f2 = torch.rand(B, 1, device=DEVICE) * 2 + 0.5
    ph = torch.rand(B, 2, device=DEVICE) * 2 * math.pi
    am = (0.35 + 0.35 * (2 * math.pi * f1 * t + ph[:, :1]).sin().abs()
               + 0.30 * (2 * math.pi * f2 * t + ph[:, 1:]).sin().abs())
    return _norm_batch(white * am, 0.65)

def synth_wind_gpu(B: int, N: int, heavy: bool = False) -> torch.Tensor:
    """Coloured noise + gust AM. HPF removes low-freq rumble."""
    base = _fft_color(B, N, 1.5)
    t    = torch.linspace(0, N / FS, N, device=DEVICE).unsqueeze(0)
    if heavy:
        gf = torch.rand(B, 1, device=DEVICE) * 0.8 + 0.3
        dp = 0.75
        base = base + torch.randn(B, N, device=DEVICE) * 0.4
    else:
        gf = torch.rand(B, 1, device=DEVICE) * 0.25 + 0.05
        dp = 0.45
    ph   = torch.rand(B, 1, device=DEVICE) * 2 * math.pi
    gust = (1 - dp) + dp * (2 * math.pi * gf * t + ph).sin().abs()
    return _norm_batch(base * gust)

def synth_pink_gpu(B: int, N: int) -> torch.Tensor:
    return _norm_batch(_fft_color(B, N, 1.0), 0.75)

_SYNTH = {
    "tank":       synth_tank_gpu,
    "engine":     synth_engine_gpu,
    "crowd":      synth_crowd_gpu,
    "wind_light": lambda B, N: synth_wind_gpu(B, N, heavy=False),
    "wind_heavy": lambda B, N: synth_wind_gpu(B, N, heavy=True),
    "pink":       synth_pink_gpu,
}
_NNAMES  = list(_NOISE_WEIGHTS.keys())
_NPROBS  = np.array([_NOISE_WEIGHTS[k] for k in _NNAMES], dtype=np.float64)
_NPROBS /= _NPROBS.sum()

def _random_noise(B: int, N: int) -> torch.Tensor:
    return _SYNTH[np.random.choice(_NNAMES, p=_NPROBS)](B, N)

def mix_snr_gpu(drone: torch.Tensor, noise: torch.Tensor,
                snr_db: float) -> torch.Tensor:
    pd    = drone.pow(2).mean(dim=1, keepdim=True).clamp(min=1e-12)
    pn    = noise.pow(2).mean(dim=1, keepdim=True).clamp(min=1e-12)
    scale = (pd / (pn * 10 ** (snr_db / 10.0))).sqrt()
    return _norm_batch(drone + scale * noise)


# ── Drone augmentation ────────────────────────────────────────────────────
def augment_drone(wav: torch.Tensor) -> torch.Tensor:
    B, N = wav.shape
    r    = torch.rand(B).numpy()
    out  = wav.clone()

    single_m = (r >= P_CLEAN) & (r < P_CLEAN + P_SINGLE_NOISE)
    multi_m  = r >= P_CLEAN + P_SINGLE_NOISE

    if single_m.any():
        n1  = _random_noise(B, N)
        snr = random.uniform(SNR_MIN, SNR_MAX)
        mx  = mix_snr_gpu(wav, n1, snr)
        out[single_m] = mx[single_m]

    if multi_m.any():
        n1   = _random_noise(B, N)
        n2   = _random_noise(B, N)
        snr1 = random.uniform(SNR_MIN, SNR_MAX)
        snr2 = random.uniform(SNR_MIN + 3, SNR_MAX)
        mx   = mix_snr_gpu(mix_snr_gpu(wav, n1, snr1), n2, snr2)
        out[multi_m] = mx[multi_m]

    return out


# ── SpecAugment ───────────────────────────────────────────────────────────
def spec_augment(x: torch.Tensor) -> torch.Tensor:
    B, C, F, T = x.shape
    fill = x.flatten(2).min(dim=2).values[:, :, None, None]
    for _ in range(2):
        f  = random.randint(0, 8)
        f0 = random.randint(0, max(0, F - f))
        x[:, :, f0:f0 + f, :] = fill
    for _ in range(2):
        t  = random.randint(0, 12)
        t0 = random.randint(0, max(0, T - t))
        x[:, :, :, t0:t0 + t] = fill
    return x


# ── Load WAV pool to GPU ───────────────────────────────────────────────────
def load_wav_pool(wav_dir: Path, label: str,
                  min_frames: int = 0) -> list:
    """
    Load all WAV files in wav_dir, apply HPF, extract 1-s windows.
    Returns list of float16 tensors [WIN_SAMPLES] on DEVICE.
    Short files (< 1 s) are TILED so all files contribute.
    """
    wavs = sorted(wav_dir.glob("*.wav"))
    if min_frames > 0:
        wavs = [f for f in wavs if sf.info(str(f)).frames >= min_frames]
    print(f"  {label}: {len(wavs):,} files", end="", flush=True)

    windows = []
    for path in wavs:
        try:
            audio, sr = sf.read(str(path), dtype="float32")
        except Exception:
            continue
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != FS:
            t     = torch.from_numpy(audio).unsqueeze(0)
            audio = FA.resample(t, sr, FS).squeeze(0).numpy()
        peak = float(np.abs(audio).max())
        if peak < 1e-4:
            continue
        audio = (audio / peak).astype(np.float32)

        # Tile files shorter than 1 s so they contribute
        if len(audio) < WIN_SAMPLES:
            reps  = (WIN_SAMPLES // len(audio)) + 2
            audio = np.tile(audio, reps)

        for s in range(0, len(audio) - WIN_SAMPLES + 1, HOP_SAMPLES):
            w = torch.from_numpy(audio[s:s + WIN_SAMPLES])
            windows.append(w.to(torch.float16).to(DEVICE))

    mb = len(windows) * WIN_SAMPLES * 2 / 1e6
    print(f"  ->  {len(windows):,} windows  ({mb:.0f} MB)")
    return windows


# ── Evaluation ────────────────────────────────────────────────────────────
def evaluate(model, drone_pool, no_drone_pool, threshold=0.5, alpha=0.4, NC=300):
    model.eval()
    DRONE_IDX = 0

    def infer_window(wav_np: np.ndarray) -> float:
        peak = float(np.abs(wav_np).max())
        if peak < 0.002:
            return 0.0
        wav_np = (wav_np / peak).astype(np.float32)
        t  = torch.from_numpy(wav_np).unsqueeze(0).to(DEVICE)
        lm = audio_to_logmel(t)     # HPF + mel inside
        with torch.no_grad():
            return torch.softmax(model(lm), dim=1)[0, DRONE_IDX].item()

    def run_chunks(gen_fn, n_chunks):
        buf    = np.zeros(WIN_SAMPLES, dtype=np.float32)
        smooth = 0.0
        hits   = 0
        total  = 0
        for i in range(n_chunks):
            chunk = gen_fn(i).cpu().float().numpy()
            buf[:HOP_SAMPLES] = buf[HOP_SAMPLES:]
            buf[HOP_SAMPLES:] = chunk
            if i < 1:
                continue
            p      = infer_window(buf.copy())
            smooth = alpha * p + (1 - alpha) * smooth
            total += 1
            if smooth >= threshold:
                hits += 1
        return 100.0 * hits / max(total, 1)

    # drone chunk generator (real WAV pool)
    _dp = [w.float().cpu().numpy() for w in
           random.sample(drone_pool, min(300, len(drone_pool)))]
    def dn(i):
        full = _dp[i % len(_dp)]
        s    = (i * HOP_SAMPLES) % max(1, len(full) - HOP_SAMPLES + 1)
        return torch.from_numpy(full[s:s + HOP_SAMPLES]).to(DEVICE)

    def tk(i):  return synth_tank_gpu(1, HOP_SAMPLES)[0]
    def eg(i):  return synth_engine_gpu(1, HOP_SAMPLES)[0]
    def wl(i):  return synth_wind_gpu(1, HOP_SAMPLES, heavy=False)[0]
    def wh(i):  return synth_wind_gpu(1, HOP_SAMPLES, heavy=True)[0]

    def mix_gen(dfn, nfn, snr):
        def g(i):
            d = dfn(i).unsqueeze(0)
            n = nfn(i).unsqueeze(0)
            return mix_snr_gpu(d, n, snr)[0]
        return g

    # real no_drone chunk generator
    _nd = [w.float().cpu().numpy() for w in
           random.sample(no_drone_pool, min(300, len(no_drone_pool)))]
    def nd_real(i):
        full = _nd[i % len(_nd)]
        s    = (i * HOP_SAMPLES) % max(1, len(full) - HOP_SAMPLES + 1)
        return torch.from_numpy(full[s:s + HOP_SAMPLES]).to(DEVICE)

    results = {
        "clean_drone":         run_chunks(dn,                            NC),
        "drone+tank_0dB":      run_chunks(mix_gen(dn, tk,  0),          NC),
        "drone+tank_-5dB":     run_chunks(mix_gen(dn, tk, -5),          NC),
        "drone+engine_0dB":    run_chunks(mix_gen(dn, eg,  0),          NC),
        "drone+wind_light":    run_chunks(mix_gen(dn, wl,  0),          NC),
        "drone+wind_heavy_0":  run_chunks(mix_gen(dn, wh,  0),          NC),
        "drone+wind_heavy-5":  run_chunks(mix_gen(dn, wh, -5),          NC),
        "tank_only_(FA)":      run_chunks(tk,                            NC),
        "engine_only_(FA)":    run_chunks(eg,                            NC),
        "wind_heavy_(FA)":     run_chunks(wh,                            NC),
        "real_no_drone_(FA)":  run_chunks(nd_real,                       NC),
    }
    model.train()
    return results


# ── Training ──────────────────────────────────────────────────────────────
def train():
    print("\n" + "=" * 62)
    print("  Phase 3b: HPF + Noise Robust Training")
    print(f"  HPF cutoff: {HPF_CUTOFF_HZ} Hz  |  Mel fmin: {F_MIN_MEL} Hz")
    print("=" * 62)

    # ── Load data ─────────────────────────────────────────────────────────
    print("\n[1/3] Loading audio pools to GPU...")
    drone_pool    = load_wav_pool(DATA_DIR / "drone",    "Drone",
                                  min_frames=WIN_SAMPLES)
    no_drone_pool = load_wav_pool(DATA_DIR / "no_drone", "No-drone",
                                  min_frames=0)

    if not drone_pool:
        sys.exit("ERROR: no drone files found")
    if not no_drone_pool:
        print("  WARNING: no real no_drone files — using only synthetic noise")

    N_DRONE    = len(drone_pool)
    N_NODRONE  = len(no_drone_pool)
    CLASSES    = ["drone", "no_drone"]
    DRONE_IDX  = 0
    NODRONE_IDX= 1

    # ── Model ─────────────────────────────────────────────────────────────
    print("\n[2/3] Building model from scratch (new feature space)...")
    model     = DroneCNN(n_classes=2).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(),
                            lr=LR_START, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=N_EPOCHS, eta_min=LR_MIN)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    save_path = MODELS_DIR / "drone_cnn_phase3b.pth"

    # ── Training loop ──────────────────────────────────────────────────────
    HALF = BATCH_SIZE // 2   # 64 drone per batch
    print(f"\n[3/3] Training {N_EPOCHS} epochs  "
          f"(drone pool={N_DRONE:,}  no_drone pool={N_NODRONE:,})\n")

    best_clean = 0.0

    for epoch in range(1, N_EPOCHS + 1):
        t0 = time.perf_counter()
        model.train()

        d_idx  = list(range(N_DRONE))
        random.shuffle(d_idx)
        nd_idx = list(range(N_NODRONE)) if N_NODRONE > 0 else []
        random.shuffle(nd_idx)
        nd_ptr = 0

        losses, n_correct, n_total = [], 0, 0

        for batch_start in range(0, N_DRONE - HALF + 1, HALF):
            # ── Drone batch ──────────────────────────────────────────
            di   = d_idx[batch_start:batch_start + HALF]
            if len(di) < HALF:
                di = (di + random.choices(d_idx, k=HALF))[:HALF]

            d_wav = torch.stack([drone_pool[i].float() for i in di])  # [HALF, WIN]
            d_wav = augment_drone(d_wav)       # noise mix on GPU
            d_mel = audio_to_logmel(d_wav)     # HPF + mel [HALF, 1, 64, T]

            # ── No-drone batch ───────────────────────────────────────
            nd_parts = []

            # Real no_drone WAVs (50% of no_drone batch)
            if N_NODRONE > 0:
                want_real = HALF // 2
                real_sel  = []
                for _ in range(want_real):
                    real_sel.append(nd_idx[nd_ptr % N_NODRONE])
                    nd_ptr += 1
                nd_wav_real  = torch.stack([no_drone_pool[i].float()
                                             for i in real_sel])  # [want_real, WIN]
                nd_mel_real  = audio_to_logmel(nd_wav_real)
                nd_parts.append(nd_mel_real)

            # Fresh GPU noise (50% of no_drone batch — hard negatives)
            want_synth = HALF - (HALF // 2 if N_NODRONE > 0 else 0)
            noise_wav  = _random_noise(want_synth, WIN_SAMPLES)
            noise_mel  = audio_to_logmel(noise_wav)
            nd_parts.append(noise_mel)

            nd_mel = torch.cat(nd_parts, dim=0)
            if nd_mel.size(0) > HALF:
                nd_mel = nd_mel[:HALF]
            elif nd_mel.size(0) < HALF:
                extra  = audio_to_logmel(_random_noise(HALF - nd_mel.size(0), WIN_SAMPLES))
                nd_mel = torch.cat([nd_mel, extra], dim=0)

            # ── Combine & SpecAugment ────────────────────────────────
            X = spec_augment(torch.cat([d_mel, nd_mel], dim=0))
            y = torch.zeros(BATCH_SIZE, dtype=torch.long, device=DEVICE)
            y[:HALF] = DRONE_IDX
            y[HALF:] = NODRONE_IDX

            # ── Optimise ─────────────────────────────────────────────
            optimizer.zero_grad(set_to_none=True)
            logits = model(X)
            loss   = criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            with torch.no_grad():
                n_correct += (logits.argmax(1) == y).sum().item()
                n_total   += BATCH_SIZE
                losses.append(loss.item())

        scheduler.step()
        ep_t   = time.perf_counter() - t0
        acc    = 100.0 * n_correct / max(n_total, 1)
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"  Epoch {epoch:3d}/{N_EPOCHS}  "
              f"loss={np.mean(losses):.4f}  acc={acc:.2f}%  "
              f"lr={lr_now:.2e}  t={ep_t:.1f}s")

        if epoch % 5 == 0 or epoch == N_EPOCHS:
            print("    [eval]", end=" ", flush=True)
            ev = evaluate(model, drone_pool, no_drone_pool, NC=300)
            for k, v in ev.items():
                tag = ("OK" if (("FA" in k and v < 5) or
                                ("FA" not in k and v > 50)) else "!!")
                print(f"{k}={v:.0f}%[{tag}]", end="  ")
            print()

            clean = ev.get("clean_drone", 0)
            if clean > best_clean:
                best_clean = clean
                torch.save({
                    "epoch":            epoch,
                    "model_state_dict": model.state_dict(),
                    "classes":          CLASSES,
                    "phase":            "3b",
                    "preprocessing": {
                        "hpf_cutoff_hz": HPF_CUTOFF_HZ,
                        "f_min_mel":     F_MIN_MEL,
                        "f_max_mel":     F_MAX_MEL,
                    },
                    "clean_drone_pct": clean,
                }, str(save_path))
                print(f"    -> saved  (clean drone {clean:.1f}%)")

    # ── Final eval ────────────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("  FINAL EVALUATION (best checkpoint)")
    print("=" * 62)
    best = torch.load(str(save_path), map_location=DEVICE, weights_only=False)
    model.load_state_dict(best["model_state_dict"])
    ev   = evaluate(model, drone_pool, no_drone_pool, NC=600)

    EXPECT = {
        "clean_drone":        True,
        "drone+tank_0dB":     True,
        "drone+tank_-5dB":    True,
        "drone+engine_0dB":   True,
        "drone+wind_light":   True,
        "drone+wind_heavy_0": True,
        "drone+wind_heavy-5": True,
        "tank_only_(FA)":     False,
        "engine_only_(FA)":   False,
        "wind_heavy_(FA)":    False,
        "real_no_drone_(FA)": False,
    }
    all_pass = True
    for k, v in ev.items():
        exp    = EXPECT.get(k, True)
        passed = (v > 50) == exp
        flag   = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"  {flag}  {k:30s}  detect={v:5.1f}%")

    print()
    if all_pass:
        print("  OVERALL: ALL PASS")
    else:
        fails = [(k, v) for k, v in ev.items()
                 if (EXPECT.get(k, True) != (v > 50))]
        print(f"  OVERALL: {len(fails)} scenario(s) need more work")
        for k, v in fails:
            print(f"    {k}: {v:.1f}%")

    print(f"\n  Model saved: {save_path}")
    _patch_scripts()


def _patch_scripts():
    """Add drone_cnn_phase3b.pth to the auto-select priority list."""
    pattern_old = '("drone_cnn_phase3.pth",'
    pattern_new = ('("drone_cnn_phase3b.pth",\n'
                   '                 "drone_cnn_phase3.pth",')
    for script in ("predict.py", "live_detector.py"):
        p = ROOT / script
        if not p.exists():
            continue
        txt = p.read_text(encoding="utf-8")
        if "phase3b" in txt:
            continue
        if pattern_old in txt:
            p.write_text(txt.replace(pattern_old, pattern_new), encoding="utf-8")
            print(f"  Patched {script}")


if __name__ == "__main__":
    train()
