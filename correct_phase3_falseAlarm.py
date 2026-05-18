"""
correct_phase3_falseAlarm.py
────────────────────────────
Lightweight false-alarm correction for drone_cnn_phase3.pth.

PROBLEM
  Phase 3 was trained to detect drone even in heavy tank noise.
  Side-effect: pure tank audio also triggers the detector (95% false alarm).

FIX
  Fine-tune ONLY the final FC layer (2 outputs, 64 inputs) of Phase 3.
  Freeze all Conv/BN layers.  Add pure-tank windows as no_drone examples.
  Train 10 epochs, batch=32, CPU-only → very fast, safe for the PC.

OUTPUT
  models/drone_cnn_phase3_corrected.pth
  (internal_test.py will pick it up automatically)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import torchaudio.transforms as T
from pathlib import Path

ROOT    = Path(__file__).parent
DEVICE  = torch.device("cpu")   # CPU only – no risk of crash
FS      = 16000
WIN_LEN = 16000
HOP_LEN = 8000

print("=" * 55)
print("  PHASE 3 FALSE-ALARM CORRECTION  (CPU fine-tune)")
print("=" * 55)

# ── Mel transform ─────────────────────────────────────────────────────────
_mel = T.MelSpectrogram(
    sample_rate=FS, n_fft=512,
    win_length=400, hop_length=160,
    n_mels=64, power=2.0,
).to(DEVICE)


# ── Model ─────────────────────────────────────────────────────────────────
class DroneCNN(nn.Module):
    def __init__(self, n_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc  = nn.Linear(64, n_classes)

    def forward(self, x):
        return self.fc(self.gap(self.features(x)).view(x.size(0), -1))


# ── Load Phase 3 ──────────────────────────────────────────────────────────
ckpt_path = ROOT / "models" / "drone_cnn_phase3.pth"
if not ckpt_path.exists():
    raise FileNotFoundError(f"Not found: {ckpt_path}")

ckpt    = torch.load(str(ckpt_path), map_location=DEVICE, weights_only=False)
classes = ckpt.get("classes", ["drone", "no_drone"])
model   = DroneCNN(len(classes)).to(DEVICE)
model.load_state_dict(ckpt["model_state_dict"])
drone_idx    = classes.index("drone")
no_drone_idx = classes.index("no_drone")
print(f"  Phase 3 loaded   drone_idx={drone_idx}   no_drone_idx={no_drone_idx}")

# ── Freeze everything except FC ───────────────────────────────────────────
for p in model.features.parameters():
    p.requires_grad = False
for p in model.gap.parameters():
    p.requires_grad = False
trainable = sum(p.numel() for p in model.fc.parameters())
print(f"  Trainable params : {trainable}  (FC layer only)")


# ── Audio helpers ──────────────────────────────────────────────────────────
def _lp(s, taps):
    return np.convolve(s, np.ones(taps) / taps, mode="same")

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
    rng = np.random.default_rng(int(t0 * 100) % 9999)
    for pos in range(0, n, int(FS * 0.15)):
        b = min(int(FS * 0.01), n - pos)
        clank[pos:pos + b] = rng.standard_normal(b) * 0.4
    return _norm(eng + clank + _lp(np.random.randn(n), 64) * 0.3)

def audio_to_logmel(wav: np.ndarray) -> torch.Tensor:
    """wav → [1, 1, n_mels, T] log-mel tensor on CPU."""
    peak = np.abs(wav).max()
    if peak < 1e-6:
        return None
    w   = wav / peak
    t   = torch.from_numpy(w.astype(np.float32)).unsqueeze(0)
    mel = _mel(t)
    return torch.log10(mel + 1e-10).unsqueeze(0)


# ── Build a small balanced dataset ─────────────────────────────────────────
import soundfile as sf

N_TANK   = 600   # tank-only windows as no_drone
N_DRONE  = 600   # real drone windows as drone

print(f"\n  Building dataset: {N_DRONE} drone + {N_TANK} tank windows ...")

# Drone files
drone_wav_paths = [
    f for f in sorted((ROOT / "data" / "raw" / "drone").glob("*.wav"))
    if sf.info(str(f)).frames >= WIN_LEN
]
if not drone_wav_paths:
    raise RuntimeError("No drone WAV files found in data/raw/drone/")

random.seed(0)
drone_lms = []
loaded_drone = 0
for _ in range(N_DRONE * 3):  # try more files in case some are too short
    f = random.choice(drone_wav_paths)
    audio, sr = sf.read(str(f), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    # resample if needed
    if sr != FS:
        import torchaudio.functional as FA
        t = torch.from_numpy(audio).unsqueeze(0)
        audio = FA.resample(t, sr, FS).squeeze(0).numpy()
    if len(audio) < WIN_LEN:
        continue
    # Random crop
    start = random.randint(0, len(audio) - WIN_LEN)
    lm = audio_to_logmel(audio[start:start + WIN_LEN])
    if lm is not None:
        drone_lms.append(lm)
        loaded_drone += 1
        if loaded_drone >= N_DRONE:
            break

# Tank-only windows
tank_lms = []
for k in range(N_TANK):
    t0  = k * WIN_LEN / FS
    wav = synth_tank(WIN_LEN, t0=t0)
    lm  = audio_to_logmel(wav)
    if lm is not None:
        tank_lms.append(lm)

print(f"  Drone windows    : {len(drone_lms)}")
print(f"  Tank  windows    : {len(tank_lms)}")

# Concatenate into tensors
X_drone = torch.cat(drone_lms, dim=0)                          # [N, 1, H, W]
X_tank  = torch.cat(tank_lms,  dim=0)

y_drone = torch.full((len(drone_lms),), drone_idx,    dtype=torch.long)
y_tank  = torch.full((len(tank_lms),),  no_drone_idx, dtype=torch.long)

X_all = torch.cat([X_drone, X_tank], dim=0)
y_all = torch.cat([y_drone, y_tank], dim=0)

# Shuffle
perm  = torch.randperm(len(X_all))
X_all = X_all[perm]
y_all = y_all[perm]

# Train / val split (80 / 20)
n_train = int(0.8 * len(X_all))
X_tr, X_va = X_all[:n_train], X_all[n_train:]
y_tr, y_va = y_all[:n_train], y_all[n_train:]
print(f"  Train={len(X_tr)}  Val={len(X_va)}")


# ── Training ───────────────────────────────────────────────────────────────
EPOCHS     = 10
BATCH_SIZE = 32
LR         = 1e-3   # higher LR to quickly move the FC weights

optimizer  = optim.Adam(model.fc.parameters(), lr=LR)
criterion  = nn.CrossEntropyLoss()
scheduler  = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.3)

model.train()
print(f"\n  Training (CPU, {EPOCHS} epochs, batch={BATCH_SIZE}) ...")

for epoch in range(EPOCHS):
    # Shuffle training data each epoch
    perm = torch.randperm(len(X_tr))
    X_tr = X_tr[perm]
    y_tr = y_tr[perm]

    running_loss = 0.0
    n_batches    = 0
    for i in range(0, len(X_tr), BATCH_SIZE):
        xb = X_tr[i:i + BATCH_SIZE]
        yb = y_tr[i:i + BATCH_SIZE]
        optimizer.zero_grad()
        out  = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        n_batches    += 1

    scheduler.step()

    # Validation accuracy
    model.eval()
    with torch.no_grad():
        logits = model(X_va)
        preds  = logits.argmax(dim=1)
        acc    = (preds == y_va).float().mean().item() * 100

        # Per-class accuracy
        drone_mask = (y_va == drone_idx)
        tank_mask  = (y_va == no_drone_idx)
        acc_drone  = (preds[drone_mask] == y_va[drone_mask]).float().mean().item() * 100 \
                     if drone_mask.any() else float('nan')
        acc_tank   = (preds[tank_mask]  == y_va[tank_mask]).float().mean().item()  * 100 \
                     if tank_mask.any()  else float('nan')

    model.train()

    avg_loss = running_loss / max(n_batches, 1)
    print(f"  Epoch {epoch+1:2d}/{EPOCHS}  loss={avg_loss:.4f}  "
          f"val_acc={acc:.1f}%  "
          f"drone={acc_drone:.1f}%  tank_reject={acc_tank:.1f}%")


# ── Save ──────────────────────────────────────────────────────────────────
model.eval()
out_path = ROOT / "models" / "drone_cnn_phase3_corrected.pth"
torch.save({
    "model_state_dict": model.state_dict(),
    "classes":          classes,
    "correction":       "FC fine-tuned to reject pure tank false alarms",
    "base_model":       "drone_cnn_phase3.pth",
}, str(out_path))
print(f"\n  Saved: {out_path}")
print("  Run internal_test.py to verify.\n")
