"""
finetune_engine.py
------------------
FC-only fine-tune to eliminate engine false alarms.
Uses improved synth_engine_v2 (heavy broadband noise, random f0).
Freezes all Conv/BN, trains only the final FC (130 params).  ~3-5 min CPU.
"""
import random
import sys
import numpy as np
import scipy.signal as ssig
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

ROOT       = Path(__file__).parent
MODELS_DIR = ROOT / "models"
DATA_DIR   = ROOT / "data"
DRONE_DIR  = DATA_DIR / "raw" / "drone"
FS         = 16000
WIN        = 16000
HOP        = 8000
N_EPOCHS   = 12
BATCH      = 64
LR         = 3e-4

import torchaudio.transforms as T
import torchaudio.functional as TAF
import soundfile as sf

_mel = T.MelSpectrogram(
    sample_rate=FS, n_fft=512, win_length=400,
    hop_length=160, n_mels=64, power=2.0)

def log_mel(x):
    x = torch.from_numpy(x).float().unsqueeze(0)   # (1, WIN)
    S = _mel(x)                                     # (1, 64, W)
    return torch.log10(S + 1e-10)                   # (1, 64, W)  -- no extra unsqueeze

# ── Filters ─────────────────────────────────────────────────────────────────
_HP150 = ssig.butter(4, 150,         btype='high', fs=FS, output='sos')
_HP250 = ssig.butter(4, 250,         btype='high', fs=FS, output='sos')
_BP200 = ssig.butter(4, [200, 6000], btype='band', fs=FS, output='sos')
_BP500 = ssig.butter(4, [500, 6000], btype='band', fs=FS, output='sos')

def make_views(x):
    x = x.astype(np.float64)
    def fv(s):
        pk = np.abs(s).max()
        return (s / (pk + 1e-9)).astype(np.float32)
    return [
        fv(x),
        fv(ssig.sosfiltfilt(_HP150, x)),
        fv(ssig.sosfiltfilt(_HP250, x)),
        fv(ssig.sosfiltfilt(_BP200, x)),
        fv(ssig.sosfiltfilt(_BP500, x)),
    ]

# ── Synthesizers ────────────────────────────────────────────────────────────
def _lp(s, taps):
    return np.convolve(s, np.ones(taps) / taps, mode='same')

def _norm(s, lv=0.85):
    p = np.abs(s).max()
    return (s / p * lv).astype(np.float32) if p > 1e-7 else s.astype(np.float32)

def synth_engine_v2(n, t0=0.0):
    """Improved engine: random f0 + heavy broadband exhaust noise."""
    rng = np.random.default_rng(int(t0 * 1000 + 17) % 99991)
    f0  = rng.uniform(60.0, 120.0)
    t   = np.linspace(t0, t0 + n / FS, n, endpoint=False)
    rpm = 1.0 + 0.05 * np.sin(2 * np.pi * 1.2 * t)
    ph  = np.cumsum(rpm) * (f0 / FS) * 2 * np.pi
    harm = (0.55 * np.sin(ph) +
            0.25 * np.sin(2 * ph) +
            0.12 * np.sin(3 * ph) +
            0.06 * np.sin(4 * ph) +
            0.03 * np.sin(5 * ph))
    exhaust = _lp(rng.standard_normal(n), max(1, int(FS / 2000))) * 0.7
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

def synth_tank(n, t0=0.0):
    t   = np.linspace(t0, t0 + n / FS, n, endpoint=False)
    rpm = 1.0 + 0.04 * np.sin(2 * np.pi * 0.3 * t)
    f0  = 45.0
    eng = (0.55 * np.sin(2 * np.pi * f0 * rpm * t) +
           0.25 * np.sin(2 * np.pi * f0 * 2 * rpm * t) +
           0.12 * np.sin(2 * np.pi * f0 * 3 * rpm * t) +
           0.08 * np.sin(2 * np.pi * f0 * 4 * rpm * t))
    rng = np.random.default_rng(int(t0 * 100) % 9999)
    clank = np.zeros(n)
    for pos in range(0, n, int(FS * 0.15)):
        b = min(int(FS * 0.01), n - pos)
        clank[pos:pos + b] = rng.standard_normal(b) * 0.4
    return _norm(eng + clank + _lp(np.random.randn(n), 64) * 0.3)

# ── Model ────────────────────────────────────────────────────────────────────
class DroneCNN(nn.Module):
    def __init__(self, n=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU())
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc  = nn.Linear(64, n)
    def forward(self, x):
        return self.fc(self.gap(self.features(x)).flatten(1))

# ── Load ─────────────────────────────────────────────────────────────────────
model_path = MODELS_DIR / "drone_cnn_phase2_v3_multiview_hardnegatives.pth"
ckpt = torch.load(model_path, map_location='cpu')
sd   = ckpt.get('model_state_dict', ckpt)
model = DroneCNN()
model.load_state_dict(sd)
classes   = ckpt.get('classes', ['drone', 'no_drone'])
drone_idx = classes.index('drone') if 'drone' in classes else 0
no_drone_idx = 1 - drone_idx
print(f"Loaded: {model_path.name}  drone_idx={drone_idx}  classes={classes}")

# Freeze everything except FC
for name, p in model.named_parameters():
    p.requires_grad = ('fc' in name)
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable params: {trainable}  (FC only)")

# ── Load drone windows ────────────────────────────────────────────────────────
print("Loading drone windows ...")
drone_wins = []
# Most drone files are < 1s -- iterate all files but skip short ones
for f in sorted(DRONE_DIR.glob("*.wav")):
    if len(drone_wins) >= 3000:
        break
    try:
        a, sr = sf.read(str(f), dtype='float32', always_2d=False)
        if a.ndim > 1:
            a = a.mean(1)
        if sr != FS:
            a = TAF.resample(torch.from_numpy(a).unsqueeze(0), sr, FS).squeeze(0).numpy()
        if len(a) < WIN:          # skip clips shorter than 1 second
            continue
        a = a - a.mean()
        pk = np.abs(a).max()
        if pk < 1e-4:
            continue
        a /= pk
        for s in range(0, len(a) - WIN + 1, HOP):
            drone_wins.append(a[s:s + WIN].copy())
            if len(drone_wins) >= 3000:
                break
    except Exception:
        pass
print(f"  Drone windows: {len(drone_wins)}")

# ── Build fine-tune dataset ───────────────────────────────────────────────────
N_ENG = 3000
N_TAN = 1500
N_DRN = min(len(drone_wins), 3000)

print(f"Building dataset: drone={N_DRN}  engine_v2={N_ENG}  tank={N_TAN} ...")
samples_x = []
samples_y = []

# drone
idxs = list(range(len(drone_wins)))
random.shuffle(idxs)
for i in idxs[:N_DRN]:
    v = random.choice(make_views(drone_wins[i]))
    samples_x.append(v)
    samples_y.append(drone_idx)

# improved engine
for k in range(N_ENG):
    seg = synth_engine_v2(WIN, k * HOP / FS)
    v   = random.choice(make_views(seg))
    samples_x.append(v)
    samples_y.append(no_drone_idx)

# tank
for k in range(N_TAN):
    seg = synth_tank(WIN, k * HOP / FS)
    v   = random.choice(make_views(seg))
    samples_x.append(v)
    samples_y.append(no_drone_idx)

print("Computing log-mel spectrograms ...")
Xs = torch.stack([log_mel(s) for s in samples_x])
Ys = torch.tensor(samples_y, dtype=torch.long)
print(f"  Tensor: {Xs.shape}")

# ── Fine-tune ─────────────────────────────────────────────────────────────────
opt  = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
crit = nn.CrossEntropyLoss()
N    = len(Xs)
perm = torch.randperm(N)
split = int(N * 0.85)
tr_X, tr_Y = Xs[perm[:split]], Ys[perm[:split]]
va_X, va_Y = Xs[perm[split:]], Ys[perm[split:]]

model.train()
print(f"\nFine-tuning FC ({N_EPOCHS} epochs, batch={BATCH}, LR={LR}) ...")
print(f"  {'Ep':>3}  {'TrLoss':>8}  {'TrAcc':>7}  {'VaLoss':>8}  {'VaAcc':>7}")
print(f"  {'-'*45}")

best_val = 1e9
best_sd  = None

for ep in range(1, N_EPOCHS + 1):
    perm2 = torch.randperm(len(tr_X))
    tl = ta = 0.0
    nb = 0
    for s in range(0, len(tr_X), BATCH):
        idx = perm2[s:s + BATCH]
        xb, yb = tr_X[idx], tr_Y[idx]
        opt.zero_grad()
        out  = model(xb)
        loss = crit(out, yb)
        loss.backward()
        opt.step()
        tl += loss.item()
        ta += (out.argmax(1) == yb).float().mean().item()
        nb += 1
    model.eval()
    with torch.no_grad():
        vo = model(va_X)
        vl = crit(vo, va_Y).item()
        va = (vo.argmax(1) == va_Y).float().mean().item()
    model.train()
    star = "*" if vl < best_val else " "
    if vl < best_val:
        best_val = vl
        best_sd  = {k: v.clone() for k, v in model.state_dict().items()}
    print(f"  {ep:3d}  {tl/nb:8.4f}  {ta/nb:7.2%}  {vl:8.4f}  {va:7.2%} {star}")

# ── Quick FA / recall check ───────────────────────────────────────────────────
model.load_state_dict(best_sd)
model.eval()
VIEW_WEIGHTS = np.array([0.05, 0.20, 0.25, 0.35, 0.15], dtype=np.float32)

def score_window(win):
    vs = make_views(win)
    probs = np.zeros(5)
    with torch.no_grad():
        for vi, v in enumerate(vs):
            lm  = log_mel(v).unsqueeze(0)
            out = torch.softmax(model(lm), 1)
            probs[vi] = out[0, drone_idx].item()
    return float(VIEW_WEIGHTS @ probs)

def filtered_max(win):
    vs = make_views(win)
    probs = np.zeros(5)
    with torch.no_grad():
        for vi, v in enumerate(vs):
            lm  = log_mel(v).unsqueeze(0)
            out = torch.softmax(model(lm), 1)
            probs[vi] = out[0, drone_idx].item()
    return float(probs[1:].max())   # views 2-5

print("\n-- Quick FA / recall check (200 windows each) --")
print(f"  {'Label':12s}  {'mean':>6}  {'max':>6}  {'FA@0.60':>8}  {'filtMax>0.75':>13}")
for label, gen in [
        ("engine_v2",  lambda k: synth_engine_v2(WIN, k * HOP / FS)),
        ("tank",       lambda k: synth_tank(WIN, k * HOP / FS)),
        ("drone",      lambda k: drone_wins[k % len(drone_wins)]),
]:
    sc = np.array([score_window(gen(k)) for k in range(200)])
    fm = np.array([filtered_max(gen(k)) for k in range(200)])
    print(f"  {label:12s}  {sc.mean():6.3f}  {sc.max():6.3f}"
          f"  {np.mean(sc > 0.60)*100:7.1f}%  {np.mean(fm > 0.75)*100:12.1f}%")

# ── Save ──────────────────────────────────────────────────────────────────────
payload = {**ckpt, 'model_state_dict': best_sd}
torch.save(payload, model_path)
print(f"\nSaved -> {model_path}")
print("Done. Run internal_test.py to verify.")
