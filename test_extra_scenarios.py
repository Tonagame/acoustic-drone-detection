"""
test_extra_scenarios.py
-----------------------
Extra scenario tests:
  - High-pitch military drone  (blade-pass 300-500 Hz, motor whine 2-4 kHz)
  - Wind noise                 (1/f turbulence, gusts, no harmonics)
  - Wind + tank                (combined interference)
  - Wind + high-pitch drone    (detection under wind)
"""
import random
import time
from collections import deque
from pathlib import Path

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
N_CHUNKS   = 1200   # 600 s simulated per scenario
SMOOTH_N   = 3
SMOOTH_K   = 2

VIEW_WEIGHTS = np.array([0.05, 0.20, 0.25, 0.35, 0.15], dtype=np.float32)
FMAX_THR, SCORE_THR, VOTE_THR, VOTES_NEED = 0.75, 0.60, 0.60, 2

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

def _lp(s, taps):
    return np.convolve(s, np.ones(taps) / taps, mode='same')

def _norm(s, lv=0.85):
    p = np.abs(s).max()
    return (s / p * lv).astype(np.float32) if p > 1e-7 else s.astype(np.float32)

def mix_snr(a, b, snr_db):
    pa = np.mean(a**2) + 1e-12
    pb = np.mean(b**2) + 1e-12
    return _norm(a + np.sqrt(pa / (pb * 10**(snr_db/10))) * b)

# ── Synthesizers ─────────────────────────────────────────────────────────────

def synth_military_drone(n, t0=0.0):
    """
    High-pitch military / tactical UAV.
    Blade-pass frequency 300-500 Hz (faster rotors than consumer drones),
    strong electric motor whine at 2-4 kHz, light propeller wash noise.
    Deliberately uses frequencies NOT in DADS commercial-drone training data
    to test model generalisation.
    """
    rng = np.random.default_rng(int(t0 * 1000 + 7) % 99991)
    bpf = rng.uniform(300.0, 500.0)          # blade-pass fundamental
    motor_f = rng.uniform(2000.0, 3500.0)    # motor electrical frequency

    t   = np.linspace(t0, t0 + n / FS, n, endpoint=False)
    rpm = 1.0 + 0.015 * np.sin(2 * np.pi * 8.0 * t)   # fast RPM wobble

    # Blade-pass harmonics (higher pitched than commercial)
    ph  = np.cumsum(rpm) * (bpf / FS) * 2 * np.pi
    blade = (0.50 * np.sin(ph) +
             0.25 * np.sin(2 * ph) +
             0.12 * np.sin(3 * ph) +
             0.07 * np.sin(4 * ph) +
             0.04 * np.sin(5 * ph) +
             0.02 * np.sin(6 * ph))

    # Motor electrical whine (narrow-band at 2-3.5 kHz)
    motor_rpm = 1.0 + 0.005 * np.sin(2 * np.pi * 12.0 * t)
    motor_ph  = np.cumsum(motor_rpm) * (motor_f / FS) * 2 * np.pi
    motor = 0.20 * np.sin(motor_ph)

    # Propeller aerodynamic wash (broad-band 500-4000 Hz)
    wash = _lp(rng.standard_normal(n), max(1, int(FS / 4000))) * 0.10
    hp   = ssig.sosfiltfilt(ssig.butter(4, 500, btype='high', fs=FS, output='sos'),
                             wash.astype(np.float64)).astype(np.float32)

    return _norm(blade + motor + hp)


def synth_wind(n, t0=0.0):
    """
    Outdoor wind / turbulence.
    Coloured (1/f) noise with:
      - Strong low-frequency rumble (< 300 Hz)
      - Occasional gusts (amplitude bursts)
      - No harmonic structure
    Should NOT trigger drone detection.
    """
    rng  = np.random.default_rng(int(t0 * 1000 + 31) % 99991)
    # 1/f noise via cumulative sum of white noise (then differentiate)
    white = rng.standard_normal(n + 100)
    pink  = np.cumsum(white)[100:]          # integrate -> 1/f^2 (brown)
    pink  = np.diff(np.concatenate([[0], pink]))  # differentiate -> 1/f (pink)

    # Low-pass at 800 Hz (wind energy concentrated below)
    lp    = ssig.sosfiltfilt(ssig.butter(4, 800, btype='low', fs=FS, output='sos'),
                              pink.astype(np.float64)).astype(np.float32)

    # Amplitude envelope: slow gusts (0.5–2 Hz AM)
    t     = np.linspace(t0, t0 + n / FS, n, endpoint=False)
    gust  = 0.6 + 0.4 * np.abs(np.sin(2 * np.pi * rng.uniform(0.5, 2.0) * t))

    return _norm(lp * gust)


def synth_tank(n, t0=0.0):
    t   = np.linspace(t0, t0 + n / FS, n, endpoint=False)
    rpm = 1.0 + 0.04 * np.sin(2 * np.pi * 0.3 * t)
    f0  = 45.0
    eng = (0.55 * np.sin(2*np.pi*f0*rpm*t) + 0.25*np.sin(2*np.pi*f0*2*rpm*t) +
           0.12 * np.sin(2*np.pi*f0*3*rpm*t) + 0.08*np.sin(2*np.pi*f0*4*rpm*t))
    rng   = np.random.default_rng(int(t0 * 100) % 9999)
    clank = np.zeros(n)
    for pos in range(0, n, int(FS * 0.15)):
        b = min(int(FS * 0.01), n - pos)
        clank[pos:pos+b] = rng.standard_normal(b) * 0.4
    return _norm(eng + clank + _lp(np.random.randn(n), 64) * 0.3)


# ── Model ─────────────────────────────────────────────────────────────────────
class DroneCNN(nn.Module):
    def __init__(self, n=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1,16,3,padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2,2),
            nn.Conv2d(16,32,3,padding=1),nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2,2),
            nn.Conv2d(32,64,3,padding=1),nn.BatchNorm2d(64), nn.ReLU())
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc  = nn.Linear(64, n)
    def forward(self, x):
        return self.fc(self.gap(self.features(x)).flatten(1))

import torchaudio.transforms as T
_mel = T.MelSpectrogram(sample_rate=FS, n_fft=512, win_length=400,
                        hop_length=160, n_mels=64, power=2.0)

def audio_to_logmel(x):
    t = torch.from_numpy(x).float().unsqueeze(0)
    return torch.log10(_mel(t) + 1e-10)   # (1, 64, W)

def load_best_model():
    for name in ('drone_cnn_phase2_v3_multiview_hardnegatives.pth',
                 'drone_cnn_phase2_v3b_engine_v2.pth',
                 'drone_cnn_phase3.pth'):
        p = MODELS_DIR / name
        if p.exists():
            ckpt = torch.load(str(p), map_location='cpu')
            sd   = ckpt.get('model_state_dict', ckpt)
            m    = DroneCNN()
            m.load_state_dict(sd)
            m.eval()
            cls  = ckpt.get('classes', ['drone','no_drone'])
            didx = cls.index('drone') if 'drone' in cls else 0
            print(f"  Model: {name}")
            return m, didx
    raise FileNotFoundError("No model found")

@torch.no_grad()
def score_window(model, drone_idx, win):
    views = make_views(win)
    probs = np.zeros(5)
    for vi, v in enumerate(views):
        lm  = audio_to_logmel(v).unsqueeze(0)
        out = torch.softmax(model(lm), 1)
        probs[vi] = out[0, drone_idx].item()
    ws  = float(VIEW_WEIGHTS @ probs)
    fm  = float(probs[1:].max())
    vc  = int((probs > VOTE_THR).sum())
    det = (fm > FMAX_THR) or (ws > SCORE_THR) or (vc >= VOTES_NEED)
    return ws, fm, det, probs

# ── Drone pool ────────────────────────────────────────────────────────────────
_dq = []
def next_drone(files):
    global _dq
    if not _dq:
        import torchaudio.functional as TAF
        f = random.choice(files)
        a, sr = sf.read(str(f), dtype='float32', always_2d=False)
        if a.ndim > 1: a = a.mean(1)
        if sr != FS:
            a = TAF.resample(torch.from_numpy(a).unsqueeze(0), sr, FS).squeeze(0).numpy()
        a = a - a.mean(); pk = np.abs(a).max()
        if pk > 1e-4: a /= pk
        for s in range(0, len(a) - HOP + 1, HOP):
            _dq.append(a[s:s+HOP].copy())
    return _dq.pop(0)

# ── Scenario runner ───────────────────────────────────────────────────────────
def run_scenario(model, drone_idx, name, expect, gen_fn, n=N_CHUNKS):
    global _dq
    _dq = []
    buf   = np.zeros(WIN, dtype=np.float32)
    sbuf  = deque([False]*SMOOTH_N, maxlen=SMOOTH_N)
    events = ws_sum = fm_sum = 0
    counted = 0
    for i in range(n):
        chunk = gen_fn(i).astype(np.float32)
        buf[:HOP] = buf[HOP:]
        buf[HOP:] = chunk
        if i < 1: continue
        ws, fm, det, probs = score_window(model, drone_idx, buf.copy())
        sbuf.append(det)
        if sum(sbuf) >= SMOOTH_K: events += 1
        ws_sum += ws; fm_sum += fm; counted += 1

    evt  = events / max(counted, 1) * 100
    ws_m = ws_sum  / max(counted, 1)
    fm_m = fm_sum  / max(counted, 1)

    if expect:
        verdict = "PASS" if evt > 50 else ("MARGINAL" if evt > 20 else "FAIL")
    else:
        verdict = "OK  " if evt <  5 else ("WARN" if evt < 15 else "FA!!")

    flag = "OK" if (expect and evt > 50) or (not expect and evt < 5) else "!!"
    print(f"  {'DETECT' if expect else 'FA-CHK'} | {name:<38} | "
          f"events={evt:5.1f}%  ws={ws_m:.3f}  fm={fm_m:.3f}  [{verdict}] {flag}")
    return evt, ws_m

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "="*75)
    print("  EXTRA SCENARIO TESTS — military drone / wind / combined")
    print("="*75)

    model, didx = load_best_model()

    drone_files = [f for f in sorted(DRONE_DIR.glob('*.wav'))
                   if sf.info(str(f)).frames >= WIN]
    print(f"  Drone pool : {len(drone_files)} files\n")

    print(f"  {'Type':<7} | {'Scenario':<38} | {'events':>8}  {'ws':>6}  {'fm':>6}  result")
    print("  " + "-"*73)

    scenarios = [
        # --- Positive (should DETECT) ---
        ("Real DADS drone (baseline)",
         True,
         lambda i: next_drone(drone_files)),

        ("Military drone — high pitch 300-500 Hz (generalisation test)",
         True,
         lambda i: synth_military_drone(HOP, i*HOP/FS)),

        ("Military drone + wind  0 dB",
         True,
         lambda i: mix_snr(synth_military_drone(HOP, i*HOP/FS),
                            synth_wind(HOP, i*HOP/FS), 0)),

        ("Military drone + wind -5 dB  (wind 3x louder)",
         True,
         lambda i: mix_snr(synth_military_drone(HOP, i*HOP/FS),
                            synth_wind(HOP, i*HOP/FS), -5)),

        ("Military drone + wind -10 dB  (wind 10x louder)",
         True,
         lambda i: mix_snr(synth_military_drone(HOP, i*HOP/FS),
                            synth_wind(HOP, i*HOP/FS), -10)),

        ("Real DADS drone + wind  0 dB",
         True,
         lambda i: mix_snr(next_drone(drone_files),
                            synth_wind(HOP, i*HOP/FS), 0)),

        # --- Negative (should NOT detect) ---
        ("Wind only  (FA check)",
         False,
         lambda i: synth_wind(HOP, i*HOP/FS)),

        ("Wind + tank  0 dB  (FA check)",
         False,
         lambda i: mix_snr(synth_wind(HOP, i*HOP/FS),
                            synth_tank(HOP, i*HOP/FS), 0)),

        ("Wind + tank -5 dB  (tank louder, FA check)",
         False,
         lambda i: mix_snr(synth_wind(HOP, i*HOP/FS),
                            synth_tank(HOP, i*HOP/FS), -5)),
    ]

    results = []
    for name, expect, gen_fn in scenarios:
        evt, ws = run_scenario(model, didx, name, expect, gen_fn)
        results.append((name, expect, evt))

    # Summary
    passed  = sum(1 for n,e,r in results if (e and r>50) or (not e and r<5))
    total   = len(results)
    print(f"\n  {'='*73}")
    print(f"  SUMMARY: {passed}/{total} scenarios passed")
    print()

    # Explain military drone result
    mil_evt = next(r for n,e,r in results if 'Military drone' in n and 'wind' not in n.lower())
    print("  KEY INSIGHT — military drone generalisation:")
    if mil_evt > 70:
        print(f"    Model detects high-pitch military drone at {mil_evt:.1f}%.")
        print("    The BPF-200-6kHz and HPF-250Hz views capture blade harmonics")
        print("    regardless of exact frequency — CNN learnt periodic harmonic")
        print("    patterns in that band, not one specific drone's frequency.")
    elif mil_evt > 30:
        print(f"    MARGINAL detection ({mil_evt:.1f}%) — model partially generalises.")
        print("    Military BPF at 300-500 Hz is outside the DADS training range.")
        print("    Adding real military UAV recordings to training would fix this.")
    else:
        print(f"    POOR detection ({mil_evt:.1f}%) — model does NOT generalise to")
        print("    high-pitch military drones. Training data is DADS commercial drones only.")
        print("    Need real military UAV recordings or targeted data augmentation.")

if __name__ == '__main__':
    main()
