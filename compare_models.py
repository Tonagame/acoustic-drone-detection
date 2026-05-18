"""
compare_models.py
-----------------
Side-by-side comparison of two (or more) drone detection models on the same
8 scenarios used by internal_test.py.

Usage:
  python compare_models.py                          # compares v3 vs v3b automatically
  python compare_models.py model_a.pth model_b.pth  # explicit model names (basenames)
"""
import sys
import random
import time
from pathlib import Path
from collections import deque

import numpy as np
import scipy.signal as ssig
import torch
import torch.nn as nn
import soundfile as sf

ROOT       = Path(__file__).parent
MODELS_DIR = ROOT / "models"
DRONE_DIR  = ROOT / "data" / "raw" / "drone"
FS         = 16000
WIN        = 16000
HOP        = 8000
N_CHUNKS   = 1200       # chunks per scenario (= 600 s simulated, same as internal_test.py)
SMOOTH_N   = 3
SMOOTH_K   = 2

VIEW_WEIGHTS = np.array([0.05, 0.20, 0.25, 0.35, 0.15], dtype=np.float32)
FMAX_THR     = 0.75
SCORE_THR    = 0.60
VOTE_THR     = 0.60
VOTES_NEED   = 2

# ── Filters ──────────────────────────────────────────────────────────────────
_HP150 = ssig.butter(4, 150,         btype='high', fs=FS, output='sos')
_HP250 = ssig.butter(4, 250,         btype='high', fs=FS, output='sos')
_BP200 = ssig.butter(4, [200, 6000], btype='band', fs=FS, output='sos')
_BP500 = ssig.butter(4, [500, 6000], btype='band', fs=FS, output='sos')

def _fv(s):
    pk = np.abs(s).max()
    return (s / (pk + 1e-9)).astype(np.float32)

def make_views(x):
    x = x.astype(np.float64)
    return [_fv(x),
            _fv(ssig.sosfiltfilt(_HP150, x)),
            _fv(ssig.sosfiltfilt(_HP250, x)),
            _fv(ssig.sosfiltfilt(_BP200, x)),
            _fv(ssig.sosfiltfilt(_BP500, x))]

# ── Synthesizers ─────────────────────────────────────────────────────────────
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
    rng   = np.random.default_rng(int(t0 * 100) % 9999)
    clank = np.zeros(n)
    for pos in range(0, n, int(FS * 0.15)):
        b = min(int(FS * 0.01), n - pos)
        clank[pos:pos + b] = rng.standard_normal(b) * 0.4
    return _norm(eng + clank + _lp(np.random.randn(n), 64) * 0.3)

def synth_engine(n, t0=0.0):
    """Engine v2 — random f0, heavy exhaust noise."""
    rng = np.random.default_rng(int(t0 * 1000 + 17) % 99991)
    f0  = rng.uniform(60.0, 120.0)
    t   = np.linspace(t0, t0 + n / FS, n, endpoint=False)
    rpm = 1.0 + 0.05 * np.sin(2 * np.pi * 1.2 * t)
    ph  = np.cumsum(rpm) * (f0 / FS) * 2 * np.pi
    harm = (0.55 * np.sin(ph) + 0.25 * np.sin(2 * ph) +
            0.12 * np.sin(3 * ph) + 0.06 * np.sin(4 * ph) + 0.03 * np.sin(5 * ph))
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

def synth_crowd(n, t0=0.0):
    white = np.random.randn(n)
    bp    = _lp(white - _lp(white, 80), 5)
    t     = np.linspace(t0, t0 + n / FS, n, endpoint=False)
    am    = 0.4 + 0.6 * np.abs(np.sin(2 * np.pi * 3.0 * t))
    return _norm(bp * am, 0.6)

def mix_snr(drone, noise, snr_db):
    pd = np.mean(drone ** 2) + 1e-12
    pn = np.mean(noise ** 2) + 1e-12
    sc = np.sqrt(pd / (pn * 10 ** (snr_db / 10.0)))
    return _norm(drone + sc * noise)

# ── Model ─────────────────────────────────────────────────────────────────────
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

import torchaudio.transforms as T
_mel_cpu = T.MelSpectrogram(sample_rate=FS, n_fft=512, win_length=400,
                             hop_length=160, n_mels=64, power=2.0)

def audio_to_logmel(x):
    t = torch.from_numpy(x).float().unsqueeze(0)
    S = _mel_cpu(t)
    return torch.log10(S + 1e-10)   # (1, 64, W)

def load_model(name):
    path = MODELS_DIR / name
    ckpt = torch.load(str(path), map_location='cpu')
    sd   = ckpt.get('model_state_dict', ckpt)
    m    = DroneCNN()
    m.load_state_dict(sd)
    m.eval()
    cls  = ckpt.get('classes', ['drone', 'no_drone'])
    didx = cls.index('drone') if 'drone' in cls else 0
    return m, didx

@torch.no_grad()
def score_window(model, drone_idx, win):
    views = make_views(win)
    probs = np.zeros(5)
    for vi, v in enumerate(views):
        lm  = audio_to_logmel(v).unsqueeze(0)  # (1, 1, 64, W)
        out = torch.softmax(model(lm), 1)
        probs[vi] = out[0, drone_idx].item()
    ws = float(VIEW_WEIGHTS @ probs)
    fm = float(probs[1:].max())
    vc = int((probs > VOTE_THR).sum())
    det = (fm > FMAX_THR) or (ws > SCORE_THR) or (vc >= VOTES_NEED)
    return ws, fm, det

# ── Drone pool ────────────────────────────────────────────────────────────────
def build_drone_pool():
    files = [f for f in sorted(DRONE_DIR.glob('*.wav'))
             if sf.info(str(f)).frames >= WIN]
    return files

_drone_q = []
_drone_files = []

def next_drone(files):
    global _drone_q, _drone_files
    if not _drone_q:
        import torchaudio.functional as TAF
        f = random.choice(files)
        a, sr = sf.read(str(f), dtype='float32', always_2d=False)
        if a.ndim > 1: a = a.mean(1)
        if sr != FS:
            a = TAF.resample(torch.from_numpy(a).unsqueeze(0), sr, FS).squeeze(0).numpy()
        a = a - a.mean(); pk = np.abs(a).max()
        if pk > 1e-4: a /= pk
        for s in range(0, len(a) - HOP + 1, HOP):
            _drone_q.append(a[s:s + HOP].copy())
    return _drone_q.pop(0)

# ── Per-scenario evaluation ───────────────────────────────────────────────────
def eval_scenario(models_list, gen_fn, n_chunks=N_CHUNKS):
    """
    Returns list of (smoothed_event_rate, mean_ws, mean_fm) for each model.
    gen_fn(i) -> HOP-length audio chunk.
    """
    n_models = len(models_list)
    bufs     = [np.zeros(WIN, dtype=np.float32) for _ in range(n_models)]
    sbufs    = [deque([False] * SMOOTH_N, maxlen=SMOOTH_N) for _ in range(n_models)]
    events   = [0] * n_models
    ws_sum   = [0.0] * n_models
    fm_sum   = [0.0] * n_models
    counted  = 0

    for i in range(n_chunks):
        chunk = gen_fn(i).astype(np.float32)
        for mi, (model, didx) in enumerate(models_list):
            bufs[mi][:HOP] = bufs[mi][HOP:]
            bufs[mi][HOP:] = chunk
            if i < 1:
                continue
            ws, fm, det = score_window(model, didx, bufs[mi].copy())
            sbufs[mi].append(det)
            if sum(sbufs[mi]) >= SMOOTH_K:
                events[mi] += 1
            ws_sum[mi] += ws
            fm_sum[mi] += fm
        if i >= 1:
            counted += 1

    results = []
    for mi in range(n_models):
        evt_rate = events[mi] / max(counted, 1) * 100
        results.append((evt_rate, ws_sum[mi] / max(counted, 1),
                                  fm_sum[mi] / max(counted, 1)))
    return results

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    if len(sys.argv) >= 3:
        model_names = sys.argv[1:]
    else:
        model_names = [
            'drone_cnn_phase2_v3_multiview_hardnegatives.pth',
            'drone_cnn_phase2_v3b_engine_v2.pth',
        ]

    # Load all models
    models_list = []
    labels      = []
    for name in model_names:
        path = MODELS_DIR / name
        if not path.exists():
            print(f"  [SKIP] {name} not found")
            continue
        m, didx = load_model(name)
        models_list.append((m, didx))
        label = name.replace('drone_cnn_', '').replace('.pth', '')
        labels.append(label)
        print(f"  Loaded: {name}")

    if len(models_list) < 1:
        print("No models found.")
        return

    drone_files = build_drone_pool()
    print(f"  Drone pool : {len(drone_files)} files")
    print()

    SCENARIOS = [
        # (name, expect_drone, gen_fn)
        ("Pure drone",
         True,  lambda i: next_drone(drone_files)),
        ("Drone + Tank  0dB",
         True,  lambda i: mix_snr(next_drone(drone_files),
                                  synth_tank(HOP, i * HOP / FS), 0)),
        ("Drone + Tank -5dB",
         True,  lambda i: mix_snr(next_drone(drone_files),
                                  synth_tank(HOP, i * HOP / FS), -5)),
        ("Drone + Tank -10dB",
         True,  lambda i: mix_snr(next_drone(drone_files),
                                  synth_tank(HOP, i * HOP / FS), -10)),
        ("Drone + Tank -20dB",
         True,  lambda i: mix_snr(next_drone(drone_files),
                                  synth_tank(HOP, i * HOP / FS), -20)),
        ("Tank only (FA)",
         False, lambda i: synth_tank(HOP, i * HOP / FS)),
        ("Engine only (FA)",
         False, lambda i: synth_engine(HOP, i * HOP / FS)),
        ("Crowd only (FA)",
         False, lambda i: synth_crowd(HOP, i * HOP / FS)),
    ]

    # Header
    col = 22
    hdr = f"  {'Scenario':<24}"
    for lb in labels:
        hdr += f"  {lb[:col]:^{col}}"
    print("=" * (26 + len(labels) * (col + 2)))
    print(f"  SIDE-BY-SIDE COMPARISON  ({N_CHUNKS} chunks = {N_CHUNKS//2}s per scenario)")
    print("=" * (26 + len(labels) * (col + 2)))
    print(hdr)
    print(f"  {'':<24}" + ("  " + "-" * col) * len(labels))

    rows = []
    for name, expect, gen_fn in SCENARIOS:
        global _drone_q
        _drone_q = []          # reset drone queue each scenario
        res = eval_scenario(models_list, gen_fn)
        row = [name, expect]
        line = f"  {name:<24}"
        for mi, (evt, ws, fm) in enumerate(res):
            if expect:
                tag = "PASS" if evt > 50 else ("MARG" if evt > 20 else "FAIL")
            else:
                tag = "OK  " if evt < 10 else "FA!!"
            cell = f"{evt:5.1f}%  ws={ws:.2f} [{tag}]"
            line += f"  {cell:^{col}}"
            row.append(evt)
        rows.append(row)
        print(line)

    print()
    print("=" * (26 + len(labels) * (col + 2)))
    print("  DELTA TABLE  (v3b minus v3  — positive = v3b better for that metric)")
    print("=" * (26 + len(labels) * (col + 2)))
    if len(labels) == 2:
        print(f"  {'Scenario':<24}  {'v3':>8}  {'v3b':>8}  {'delta':>8}  {'winner':>8}")
        print(f"  {'-'*60}")
        for row in rows:
            name, expect = row[0], row[1]
            v3, v3b = row[2], row[3]
            delta = v3b - v3
            if expect:
                # Higher is better for detection
                winner = "v3b" if delta > 1 else ("v3 " if delta < -1 else "tie")
            else:
                # Lower is better for FA
                winner = "v3b" if delta < -1 else ("v3 " if delta > 1 else "tie")
                delta  = -delta   # flip sign: negative FA delta = v3b better
            print(f"  {name:<24}  {v3:8.1f}%  {v3b:8.1f}%  {delta:+8.1f}pp  {winner:>8}")
    print()

if __name__ == '__main__':
    main()
