"""
train_phase3_gpu.py  --  Phase 3: Noise + Wind Robust Drone Detector

What's new vs Phase 2v2:
  1. All drone audio pre-loaded to GPU at start (zero CPU during training)
  2. ALL noise synthesis on GPU (FFT-based colored noise, harmonic engines)
  3. Online augmentation: 6 noise types, SNR -15..+10 dB, 30% multi-noise
  4. Hard negatives: pure synthetic noise batches as no_drone class
  5. Fine-tunes from drone_cnn_phase2v2.pth for fast convergence
  6. Target: >85% clean drone, >70% drone+tank 0dB, >60% drone+heavy wind

Run:
    python train_phase3_gpu.py
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
CACHE_DIR  = ROOT / "cache"
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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
if DEVICE.type != "cuda":
    print("WARNING: CUDA not found, will be slow on CPU")

# ── Hyperparameters ───────────────────────────────────────────────────────
N_EPOCHS     = 30
BATCH_SIZE   = 128    # 64 drone + 64 no-drone per batch
LR_START     = 5e-5   # fine-tune (lower than scratch)
LR_MIN       = 1e-6
WEIGHT_DECAY = 1e-4

# Augmentation mix for each drone sample
P_CLEAN       = 0.20   # keep raw drone
P_SINGLE_NOISE= 0.50   # mix with one noise type
P_MULTI_NOISE = 0.30   # mix with two noise types layered

SNR_MIN = -10.0
SNR_MAX = +10.0

# Noise type sampling weights
_NOISE_WEIGHTS = {
    "tank":       0.22,
    "engine":     0.18,
    "crowd":      0.10,
    "wind_light": 0.20,
    "wind_heavy": 0.20,
    "pink":       0.10,
}

# ── Mel transform (on GPU) ────────────────────────────────────────────────
_mel_tf = T.MelSpectrogram(
    sample_rate=FS, n_fft=N_FFT,
    win_length=WIN_LEN, hop_length=HOP_LEN,
    n_mels=N_MELS, power=2.0,
).to(DEVICE)

def audio_to_logmel(wav: torch.Tensor) -> torch.Tensor:
    """wav: [B, WIN_SAMPLES] float32 -> [B, 1, 64, T] log-mel on GPU."""
    mel = _mel_tf(wav)                       # [B, 64, T]
    lm  = torch.log10(mel + 1e-10)          # [B, 64, T]
    return lm.unsqueeze(1)                   # [B, 1, 64, T]


# ── Model (same DroneCNN = compatible with all scripts) ───────────────────
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


# ── GPU Noise Synthesizers (fully vectorized, no Python loops) ────────────

def _norm_batch(x: torch.Tensor, level: float = 0.85) -> torch.Tensor:
    peak = x.abs().amax(dim=1, keepdim=True).clamp(min=1e-7)
    return x / peak * level

def _fft_color(B: int, N: int, exponent: float) -> torch.Tensor:
    """Generate colored noise via FFT shaping: power ~ 1/f^exponent."""
    white  = torch.randn(B, N, device=DEVICE)
    X      = torch.fft.rfft(white)
    freqs  = torch.fft.rfftfreq(N, device=DEVICE).clamp(min=1e-6)
    X      = X / freqs.pow(exponent * 0.5).unsqueeze(0)
    sig    = torch.fft.irfft(X, n=N)
    return sig - sig.mean(dim=1, keepdim=True)

def synth_tank_gpu(B: int, N: int) -> torch.Tensor:
    """Diesel engine: 45 Hz harmonics + mechanical clank + low rumble."""
    t   = torch.linspace(0, N / FS, N, device=DEVICE)
    rpm = 1.0 + 0.04 * torch.sin(2 * math.pi * 0.3 * t)     # slow RPM drift
    f0  = 45.0
    eng = (0.55 * torch.sin(2 * math.pi * f0 * 1 * rpm * t) +
           0.25 * torch.sin(2 * math.pi * f0 * 2 * rpm * t) +
           0.12 * torch.sin(2 * math.pi * f0 * 3 * rpm * t) +
           0.08 * torch.sin(2 * math.pi * f0 * 4 * rpm * t))
    eng   = eng.unsqueeze(0).expand(B, -1)
    rumble = _fft_color(B, N, 1.0) * 0.3                      # low-freq rumble
    noise  = torch.randn(B, N, device=DEVICE) * 0.08          # mechanical noise
    return _norm_batch(eng + rumble + noise)

def synth_engine_gpu(B: int, N: int) -> torch.Tensor:
    """Vehicle engine: 60-100 Hz randomised per sample."""
    t   = torch.linspace(0, N / FS, N, device=DEVICE)
    rpm = 1.0 + 0.03 * torch.sin(2 * math.pi * 1.5 * t)
    f0  = (torch.randint(60, 100, (B, 1), device=DEVICE).float())  # [B,1]
    t_  = t.unsqueeze(0)                                            # [1,N]
    s   = (0.50 * torch.sin(2 * math.pi * f0 * 1 * rpm * t_) +
           0.30 * torch.sin(2 * math.pi * f0 * 2 * rpm * t_) +
           0.12 * torch.sin(2 * math.pi * f0 * 3 * rpm * t_) +
           0.08 * torch.sin(2 * math.pi * f0 * 4 * rpm * t_))
    return _norm_batch(s + torch.randn(B, N, device=DEVICE) * 0.1)

def synth_crowd_gpu(B: int, N: int) -> torch.Tensor:
    """Crowd / street: white noise with speech-rhythm amplitude modulation."""
    white = torch.randn(B, N, device=DEVICE)
    t  = torch.linspace(0, N / FS, N, device=DEVICE).unsqueeze(0)
    f1 = torch.rand(B, 1, device=DEVICE) * 4 + 3    # 3-7 Hz (syllable rate)
    f2 = torch.rand(B, 1, device=DEVICE) * 2 + 0.5  # 0.5-2.5 Hz (word/breath)
    ph = torch.rand(B, 2, device=DEVICE) * 2 * math.pi
    am = (0.35 + 0.35 * (2 * math.pi * f1 * t + ph[:, :1]).sin().abs()
               + 0.30 * (2 * math.pi * f2 * t + ph[:, 1:]).sin().abs())
    return _norm_batch(white * am, 0.65)

def synth_wind_gpu(B: int, N: int, heavy: bool = False) -> torch.Tensor:
    """
    Wind noise: 1/f^1.5 (low-freq heavy) + amplitude gusts.
    heavy=True -> stronger gusts (0.3-1.1 Hz), more turbulence.
    """
    # Colored noise: steeper than pink for wind rumble
    base   = _fft_color(B, N, 1.5)
    t      = torch.linspace(0, N / FS, N, device=DEVICE).unsqueeze(0)
    if heavy:
        gf = torch.rand(B, 1, device=DEVICE) * 0.8 + 0.3   # 0.3-1.1 Hz gusts
        dp = 0.75
        # Add turbulence bursts
        turb = torch.randn(B, N, device=DEVICE) * 0.4
        base = base + turb
    else:
        gf = torch.rand(B, 1, device=DEVICE) * 0.25 + 0.05  # 0.05-0.30 Hz
        dp = 0.45
    ph   = torch.rand(B, 1, device=DEVICE) * 2 * math.pi
    gust = (1 - dp) + dp * (2 * math.pi * gf * t + ph).sin().abs()
    return _norm_batch(base * gust)

def synth_pink_gpu(B: int, N: int) -> torch.Tensor:
    """Generic pink noise (1/f power spectrum)."""
    return _norm_batch(_fft_color(B, N, 1.0), 0.75)

# dispatch table
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


# ── SNR mixing (GPU, vectorized) ──────────────────────────────────────────
def mix_snr_gpu(drone: torch.Tensor, noise: torch.Tensor,
                snr_db: float) -> torch.Tensor:
    """drone, noise: [B, N]; returns normalised mixture at snr_db."""
    pd    = drone.pow(2).mean(dim=1, keepdim=True).clamp(min=1e-12)
    pn    = noise.pow(2).mean(dim=1, keepdim=True).clamp(min=1e-12)
    scale = (pd / (pn * 10 ** (snr_db / 10.0))).sqrt()
    return _norm_batch(drone + scale * noise)


# ── Augment a drone batch (GPU, single tensor op) ─────────────────────────
def augment_drone(wav: torch.Tensor) -> torch.Tensor:
    """
    wav: [B, WIN_SAMPLES] float32 on GPU.
    Returns: augmented batch (same shape).
    """
    B, N = wav.shape
    r   = torch.rand(B).numpy()
    out = wav.clone()

    # masks
    clean_m  = r < P_CLEAN
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
        snr2 = random.uniform(SNR_MIN + 3, SNR_MAX)   # second noise a bit quieter
        mx   = mix_snr_gpu(mix_snr_gpu(wav, n1, snr1), n2, snr2)
        out[multi_m] = mx[multi_m]

    return out


# ── SpecAugment ───────────────────────────────────────────────────────────
def spec_augment(x: torch.Tensor) -> torch.Tensor:
    """x: [B, 1, F, T]. Time + freq masking."""
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


# ── Load all drone audio to GPU ───────────────────────────────────────────
def load_drone_audio_to_gpu():
    """
    Load every drone WAV >= 1 s, extract 1-s windows, store as float16 on GPU.
    Returns: list of Tensor [WIN_SAMPLES] (float16, DEVICE).
    """
    drone_dir = DATA_DIR / "drone"
    wavs = [f for f in sorted(drone_dir.glob("*.wav"))
            if sf.info(str(f)).frames >= WIN_SAMPLES]
    print(f"  {len(wavs):,} drone files >= 1s found")

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
        for s in range(0, len(audio) - WIN_SAMPLES + 1, HOP_SAMPLES):
            w = torch.from_numpy(audio[s:s + WIN_SAMPLES])
            windows.append(w.to(torch.float16).to(DEVICE))

    mb = len(windows) * WIN_SAMPLES * 2 / 1e6
    print(f"  -> {len(windows):,} drone windows  ({mb:.1f} MB on GPU)")
    return windows


# ── Load no-drone features from Phase 2v2 cache ───────────────────────────
def load_nodrone_cache():
    """
    Returns:
      X_bg      numpy mmap [N, 64, T] float16  -- DADS no-drone mel features
      y_bg      numpy array [N]                 -- labels (all nodrone_idx)
      X_speech  numpy mmap [M, 64, T] float16  -- LibriSpeech mel features
      nodrone_idx int
    """
    X_bg = X_speech = None
    nodrone_idx = 1   # default

    aug_X = CACHE_DIR / "aug_train_X.npy"
    aug_y = CACHE_DIR / "aug_train_y.npy"
    if aug_X.exists() and aug_y.exists():
        print("  Loading DADS no-drone features from Phase 1b cache...")
        X_all = np.load(str(aug_X), mmap_mode="r")
        y_all = np.load(str(aug_y))
        # detect which label index is no_drone
        try:
            ckpt = torch.load(str(MODELS_DIR / "drone_cnn_phase2v2.pth"),
                              map_location="cpu", weights_only=False)
            cls = ckpt.get("classes", ["drone", "no_drone"])
            nodrone_idx = cls.index("no_drone")
        except Exception:
            nodrone_idx = 1
        mask  = (y_all == nodrone_idx)
        X_bg  = X_all[mask]
        print(f"  -> {len(X_bg):,} DADS no-drone windows")

    sp = CACHE_DIR / "librispeech_X.npy"
    if sp.exists():
        print("  Loading LibriSpeech speech features from Phase 2v2 cache...")
        X_speech = np.load(str(sp), mmap_mode="r")
        print(f"  -> {len(X_speech):,} speech windows")

    return X_bg, X_speech, nodrone_idx


# ── Evaluation ────────────────────────────────────────────────────────────
def evaluate(model, drone_windows, X_bg, X_speech, nodrone_idx,
             n_eval=500, alpha=0.4, threshold=0.5):
    """
    Runs the sliding-window EMA simulation on held-out drone audio.
    Returns dict: scenario -> detection_rate (%).
    """
    model.eval()
    CLASSES = ["drone", "no_drone"]
    DRONE_IDX = CLASSES.index("drone")

    def infer_window(wav_np):
        peak = float(np.abs(wav_np).max())
        if peak < 0.002:
            return 0.0
        wav_np = wav_np / peak
        t   = torch.from_numpy(wav_np).unsqueeze(0).to(DEVICE)
        lm  = audio_to_logmel(t)
        with torch.no_grad():
            p = torch.softmax(model(lm), dim=1)[0, DRONE_IDX].item()
        return p

    def run_chunks(gen_fn, n_chunks=200):
        buf    = np.zeros(WIN_SAMPLES, dtype=np.float32)
        smooth = 0.0
        probs  = []
        for i in range(n_chunks):
            chunk = gen_fn(i).cpu().numpy().astype(np.float32)
            if len(chunk) == HOP_SAMPLES:
                buf[:HOP_SAMPLES] = buf[HOP_SAMPLES:]
                buf[HOP_SAMPLES:] = chunk
            else:
                buf[:] = chunk[:WIN_SAMPLES]
            if i < 1:
                continue
            p      = infer_window(buf.copy())
            smooth = alpha * p + (1 - alpha) * smooth
            probs.append(smooth)
        return 100.0 * sum(1 for p in probs if p >= threshold) / max(len(probs), 1)

    # drone pool for eval
    eval_pool = [w.float().cpu().numpy() for w in
                 random.sample(drone_windows, min(200, len(drone_windows)))]
    epool_q   = list(eval_pool) * 10   # repeat so we don't run out

    def next_drone_chunk(i):
        idx  = i % len(eval_pool)
        full = eval_pool[idx]
        s    = (i * HOP_SAMPLES) % max(1, len(full) - HOP_SAMPLES)
        return torch.from_numpy(full[s:s + HOP_SAMPLES]).to(DEVICE)

    def dn(i):  return next_drone_chunk(i)
    def tk(i):  return synth_tank_gpu(1, HOP_SAMPLES)[0]
    def eg(i):  return synth_engine_gpu(1, HOP_SAMPLES)[0]
    def wl(i):  return synth_wind_gpu(1, HOP_SAMPLES, heavy=False)[0]
    def wh(i):  return synth_wind_gpu(1, HOP_SAMPLES, heavy=True)[0]

    def mixed(drone_fn, noise_fn, snr):
        def gen(i):
            d = drone_fn(i).unsqueeze(0)
            n = noise_fn(i).unsqueeze(0)
            return mix_snr_gpu(d, n, snr)[0]
        return gen

    NC = 300
    results = {
        "clean_drone":        run_chunks(dn,                              NC),
        "drone+tank_0dB":     run_chunks(mixed(dn, tk, 0),               NC),
        "drone+tank_-5dB":    run_chunks(mixed(dn, tk, -5),              NC),
        "drone+wind_light":   run_chunks(mixed(dn, wl, 0),               NC),
        "drone+wind_heavy_0": run_chunks(mixed(dn, wh, 0),               NC),
        "drone+wind_heavy-5": run_chunks(mixed(dn, wh, -5),              NC),
        "tank_only_(FA)":     run_chunks(tk,                              NC),
        "wind_heavy_(FA)":    run_chunks(wh,                              NC),
    }
    model.train()
    return results


# ── Training ──────────────────────────────────────────────────────────────
def train():
    print("\n" + "=" * 60)
    print("  Phase 3: Noise + Wind Robust Training")
    print("=" * 60)

    # ── Load data ────────────────────────────────────────────────────────
    print("\n[1/4] Loading drone audio to GPU...")
    drone_windows = load_drone_audio_to_gpu()
    if not drone_windows:
        sys.exit("ERROR: no drone files found in data/raw/drone/")
    N_DRONE = len(drone_windows)

    print("\n[2/4] Loading no-drone cache...")
    X_bg, X_speech, NODRONE_IDX = load_nodrone_cache()
    DRONE_IDX = 1 - NODRONE_IDX    # 0 or 1

    has_bg     = X_bg is not None and len(X_bg) > 0
    has_speech = X_speech is not None and len(X_speech) > 0
    print(f"  DRONE_IDX={DRONE_IDX}  NODRONE_IDX={NODRONE_IDX}")

    # ── Build model ───────────────────────────────────────────────────────
    print("\n[3/4] Setting up model...")
    CLASSES = ["drone", "no_drone"]
    model   = DroneCNN(n_classes=2).to(DEVICE)

    p2_path = MODELS_DIR / "drone_cnn_phase2v2.pth"
    if p2_path.exists():
        ckpt = torch.load(str(p2_path), map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"  Loaded Phase 2v2 weights from {p2_path.name}")
    else:
        print("  WARNING: Phase 2v2 model not found, training from scratch")

    optimizer = optim.AdamW(model.parameters(),
                            lr=LR_START, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=N_EPOCHS, eta_min=LR_MIN)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    # ── Training loop ─────────────────────────────────────────────────────
    print(f"\n[4/4] Training for {N_EPOCHS} epochs  "
          f"(batch={BATCH_SIZE}, drone_pool={N_DRONE:,})\n")

    HALF = BATCH_SIZE // 2      # drone per batch
    Q1   = BATCH_SIZE // 4      # no-drone from bg cache
    Q2   = BATCH_SIZE // 4      # no-drone: speech + fresh GPU noise
    # Q2 split: half speech, half fresh noise
    Q_SP = Q2 // 2
    Q_NS = Q2 - Q_SP

    best_acc  = 0.0
    save_path = MODELS_DIR / "drone_cnn_phase3.pth"

    for epoch in range(1, N_EPOCHS + 1):
        t_ep  = time.perf_counter()
        model.train()

        # Shuffle drone indices
        drone_idx_list = list(range(N_DRONE))
        random.shuffle(drone_idx_list)

        losses, n_correct, n_total = [], 0, 0
        steps = 0

        for batch_start in range(0, N_DRONE - HALF + 1, HALF):
            # ── Sample drone windows ──────────────────────────────────
            d_idx = drone_idx_list[batch_start:batch_start + HALF]
            if len(d_idx) < HALF:
                d_idx = (d_idx + random.choices(drone_idx_list, k=HALF))[:HALF]

            # float16 on GPU -> float32 for processing
            d_wav = torch.stack([drone_windows[i].float() for i in d_idx])  # [HALF, WIN]

            # ── Noise augmentation on GPU ─────────────────────────────
            d_wav = augment_drone(d_wav)   # [HALF, WIN]

            # ── Compute mel for drone ─────────────────────────────────
            d_mel = audio_to_logmel(d_wav)  # [HALF, 1, 64, T]

            # ── No-drone: background cache → GPU ──────────────────────
            nd_parts = []
            if has_bg and Q1 > 0:
                idx_bg  = np.random.randint(0, len(X_bg), Q1)
                x_bg_np = X_bg[idx_bg].astype(np.float32)
                x_bg_t  = torch.from_numpy(x_bg_np).unsqueeze(1).to(DEVICE)
                nd_parts.append(x_bg_t)

            # ── No-drone: speech ──────────────────────────────────────
            if has_speech and Q_SP > 0:
                idx_sp  = np.random.randint(0, len(X_speech), Q_SP)
                x_sp_np = X_speech[idx_sp].astype(np.float32)
                x_sp_t  = torch.from_numpy(x_sp_np).unsqueeze(1).to(DEVICE)
                nd_parts.append(x_sp_t)

            # ── No-drone: fresh GPU noise (hard negatives) ────────────
            if Q_NS > 0:
                noise_wav = _random_noise(Q_NS, WIN_SAMPLES)          # [Q_NS, WIN]
                noise_mel = audio_to_logmel(noise_wav)                 # [Q_NS, 1, 64, T]
                nd_parts.append(noise_mel)

            if not nd_parts:
                # fallback: generate all no-drone as noise
                noise_wav = _random_noise(HALF, WIN_SAMPLES)
                nd_mel    = audio_to_logmel(noise_wav)
                nd_parts  = [nd_mel]

            nd_mel = torch.cat(nd_parts, dim=0)    # [~HALF, 1, 64, T]
            # trim/pad to exactly HALF
            if nd_mel.size(0) > HALF:
                nd_mel = nd_mel[:HALF]
            elif nd_mel.size(0) < HALF:
                extra  = _random_noise(HALF - nd_mel.size(0), WIN_SAMPLES)
                nd_mel = torch.cat([nd_mel, audio_to_logmel(extra)], dim=0)

            # ── Combine & SpecAugment ─────────────────────────────────
            X_batch = torch.cat([d_mel, nd_mel], dim=0)   # [BATCH, 1, 64, T]
            X_batch = spec_augment(X_batch)

            y_batch = torch.zeros(BATCH_SIZE, dtype=torch.long, device=DEVICE)
            y_batch[:HALF] = DRONE_IDX    # drone labels
            y_batch[HALF:] = NODRONE_IDX  # no-drone labels

            # ── Forward / backward ────────────────────────────────────
            optimizer.zero_grad(set_to_none=True)
            logits = model(X_batch)
            loss   = criterion(logits, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            with torch.no_grad():
                preds     = logits.argmax(dim=1)
                n_correct += (preds == y_batch).sum().item()
                n_total   += BATCH_SIZE
                losses.append(loss.item())
            steps += 1

        scheduler.step()
        acc    = 100.0 * n_correct / max(n_total, 1)
        ep_t   = time.perf_counter() - t_ep
        lr_now = optimizer.param_groups[0]["lr"]

        print(f"  Epoch {epoch:3d}/{N_EPOCHS}  "
              f"loss={np.mean(losses):.4f}  "
              f"acc={acc:.2f}%  "
              f"lr={lr_now:.2e}  "
              f"t={ep_t:.1f}s")

        # Quick eval every 5 epochs and last epoch
        if epoch % 5 == 0 or epoch == N_EPOCHS:
            print("    [eval]", end=" ", flush=True)
            ev = evaluate(model, drone_windows, X_bg, X_speech, NODRONE_IDX)
            for k, v in ev.items():
                tag = "OK" if (("FA" in k and v < 5) or ("FA" not in k and v > 50)) else "!!"
                print(f"{k}={v:.0f}%[{tag}]", end="  ")
            print()

            clean_dr = ev.get("clean_drone", 0)
            if clean_dr > best_acc:
                best_acc = clean_dr
                torch.save({
                    "epoch":            epoch,
                    "model_state_dict": model.state_dict(),
                    "classes":          CLASSES,
                    "phase":            3,
                    "clean_drone_pct":  clean_dr,
                }, str(save_path))
                print(f"    -> saved  (clean drone {clean_dr:.1f}%)")

    # ── Final evaluation ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  FINAL EVALUATION (best checkpoint)")
    print("=" * 60)
    best_ckpt = torch.load(str(save_path), map_location=DEVICE, weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])
    ev = evaluate(model, drone_windows, X_bg, X_speech, NODRONE_IDX,
                  n_eval=600, threshold=0.5)

    scenarios_should_detect = {
        "clean_drone":        True,
        "drone+tank_0dB":     True,
        "drone+tank_-5dB":    True,
        "drone+wind_light":   True,
        "drone+wind_heavy_0": True,
        "drone+wind_heavy-5": True,
        "tank_only_(FA)":     False,
        "wind_heavy_(FA)":    False,
    }
    all_pass = True
    for k, v in ev.items():
        expect  = scenarios_should_detect.get(k, True)
        pass_ok = (v > 50) == expect
        flag    = "PASS" if pass_ok else "FAIL"
        if not pass_ok:
            all_pass = False
        print(f"  {flag}  {k:30s}  detect={v:5.1f}%")

    print()
    print(f"  OVERALL: {'ALL PASS' if all_pass else 'SOME SCENARIOS NEED MORE WORK'}")
    print(f"  Model saved: {save_path}")
    print()

    # ── Update auto-selector in predict.py + live_detector.py ────────────
    _patch_best_model()


def _patch_best_model():
    """
    Insert drone_cnn_phase3.pth into the priority list in predict.py
    and live_detector.py so they pick it up automatically.
    """
    pattern_old = '("drone_cnn_phase2v2.pth",'
    pattern_new = '("drone_cnn_phase3.pth",\n                 "drone_cnn_phase2v2.pth",'
    for script in ("predict.py", "live_detector.py"):
        p = ROOT / script
        if not p.exists():
            continue
        txt = p.read_text(encoding="utf-8")
        if "phase3" in txt:
            continue   # already patched
        if pattern_old in txt:
            p.write_text(txt.replace(pattern_old, pattern_new), encoding="utf-8")
            print(f"  Patched {script} to auto-select phase3 model")


if __name__ == "__main__":
    train()
