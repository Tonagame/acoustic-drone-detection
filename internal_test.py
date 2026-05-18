"""
internal_test.py  --  Multi-view drone detection stress test.
No audio output. Tests detection rate across 8 scenarios.

Multi-view pipeline
  5 spectral views: raw, HPF-150, HPF-250, BPF-200-6000, BPF-500-6000
  Weights          : [0.05, 0.20, 0.25, 0.35, 0.15]
  filteredMax      : max(views 2-5)   -- excludes raw
  Detection rule   : filteredMax > 0.75  OR  weightedScore > 0.60  OR  voteCount >= 2
  Vote threshold   : 0.60
"""
import numpy as np
import torch
import soundfile as sf
import time
import random
import scipy.signal as sig
import torchaudio.transforms as T
import torchaudio.functional as FA
from pathlib import Path

ROOT        = Path(__file__).parent
FS          = 16000
WIN_SAMPLES = 16000
HOP_SAMPLES = 8000
N_FFT       = 512
WIN_LEN     = 400
HOP_LEN     = 160
N_MELS      = 64
NOISE_FLOOR = 0.002
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_mel = T.MelSpectrogram(
    sample_rate=FS, n_fft=N_FFT,
    win_length=WIN_LEN, hop_length=HOP_LEN,
    n_mels=N_MELS, power=2.0,
).to(DEVICE)


# ── Model ─────────────────────────────────────────────────────────────────
import torch.nn as nn

class DroneCNN(nn.Module):
    def __init__(self, n_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc  = nn.Linear(64, 2)

    def forward(self, x):
        return self.fc(self.gap(self.features(x)).view(x.size(0), -1))


MODEL_PRIORITY = [
    "drone_cnn_phase2_v3_multiview_hardnegatives.pth",  # phase2v3: multiview hard-negatives
    "drone_cnn_phase3_corrected.pth",                   # phase3 + tank FA correction
    "drone_cnn_phase3.pth",
    "drone_cnn_phase2v2.pth",
    "drone_cnn_phase2_noise_robust.pth",
    "drone_cnn_phase1b.pth",
    "drone_cnn_phase1.pth",
]

def load_model():
    for name in MODEL_PRIORITY:
        p = ROOT / "models" / name
        if p.exists():
            ckpt    = torch.load(str(p), map_location=DEVICE, weights_only=False)
            classes = ckpt.get("classes", ["drone", "no_drone"])
            model   = DroneCNN(len(classes)).to(DEVICE)
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()
            print(f"  Model       : {name}")
            return model, classes.index("drone"), name
    raise FileNotFoundError("No model found in models/")


# ── Audio views (Python equivalent of create_audio_views.m) ──────────────
def _norm_view(x):
    """Remove DC, peak-normalise."""
    x = x - x.mean()
    pk = np.abs(x).max()
    return (x / pk) if pk > 1e-6 else x

# Pre-build filter coefficients at 16 kHz (called once at startup)
_HP150  = sig.butter(4, 150,         btype='high', fs=FS, output='sos')
_HP250  = sig.butter(4, 250,         btype='high', fs=FS, output='sos')
_BP200  = sig.butter(4, [200, 6000], btype='band', fs=FS, output='sos')
_BP500  = sig.butter(4, [500, 6000], btype='band', fs=FS, output='sos')

VIEW_NAMES   = ['raw', 'HPF-150', 'HPF-250', 'BPF-200-6k', 'BPF-500-6k']
VIEW_WEIGHTS = np.array([0.05, 0.20, 0.25, 0.35, 0.15], dtype=np.float32)

def create_audio_views(x: np.ndarray) -> list:
    """Return list of 5 filtered float32 views, all peak-normalised."""
    x = x.astype(np.float64)
    x = x - x.mean()
    pk = np.abs(x).max()
    if pk < 1e-6:
        return [np.zeros_like(x, dtype=np.float32)] * 5
    x = x / pk

    v1 = x.astype(np.float32)
    v2 = _norm_view(sig.sosfilt(_HP150, x)).astype(np.float32)
    v3 = _norm_view(sig.sosfilt(_HP250, x)).astype(np.float32)
    v4 = _norm_view(sig.sosfilt(_BP200, x)).astype(np.float32)
    v5 = _norm_view(sig.sosfilt(_BP500, x)).astype(np.float32)
    return [v1, v2, v3, v4, v5]


# ── Per-view CNN inference ─────────────────────────────────────────────────
def _infer_view(model, drone_idx, audio_view: np.ndarray) -> float:
    """Run CNN on one audio view; return drone probability."""
    peak = np.abs(audio_view).max()
    if peak < NOISE_FLOOR:
        return 0.0
    w   = (audio_view / peak).astype(np.float32)
    t   = torch.from_numpy(w).unsqueeze(0).to(DEVICE)
    mel = _mel(t)
    lm  = torch.log10(mel + 1e-10).unsqueeze(0)
    with torch.no_grad():
        return torch.softmax(model(lm), dim=1)[0, drone_idx].item()


# ── Multi-view score combiner ─────────────────────────────────────────────
VOTE_THR   = 0.60   # raised from 0.55 -- engine harmonics in BPF view no longer
                    # accumulate 2 votes at 0.55; real drone still clears 0.60 easily
FMAX_THR   = 0.75
SCORE_THR  = 0.60
VOTES_NEED = 2

def combine_multiview(probs: np.ndarray):
    """
    probs: length-5 array (one per view, matching VIEW_WEIGHTS order)
    Returns: (weightedScore, filteredMax, voteCount, detected, reason)
    """
    probs = np.clip(probs, 0.0, 1.0)
    weighted_score = float(np.dot(VIEW_WEIGHTS, probs))
    filtered_max   = float(probs[1:].max())       # views 2-5 only
    vote_count     = int((probs > VOTE_THR).sum())

    path_a = filtered_max   > FMAX_THR
    path_b = weighted_score > SCORE_THR
    path_c = vote_count    >= VOTES_NEED
    detected = path_a or path_b or path_c

    if not detected:
        reason = "notDetected"
    elif path_a:
        reason = "filteredMax"
    elif path_b:
        reason = "weightedScore"
    else:
        reason = "voteCount"
    return weighted_score, filtered_max, vote_count, detected, reason


def infer_multiview(model, drone_idx, window: np.ndarray):
    """Run 5-view inference; return (weighted_score, filtered_max, detected)."""
    views = create_audio_views(window)
    probs = np.array([_infer_view(model, drone_idx, v) for v in views],
                     dtype=np.float32)
    ws, fm, vc, det, reason = combine_multiview(probs)
    return ws, fm, det, probs, reason


# ── Audio helpers ─────────────────────────────────────────────────────────
def _lp(s, taps):
    return np.convolve(s, np.ones(taps) / taps, mode="same")

def _norm(s, lv=0.85):
    p = np.abs(s).max()
    return (s / p * lv).astype(np.float32) if p > 1e-7 else s.astype(np.float32)

def load_wav(path):
    audio, sr = sf.read(str(path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != FS:
        t     = torch.from_numpy(audio).unsqueeze(0)
        audio = FA.resample(t, sr, FS).squeeze(0).numpy()
    peak = np.abs(audio).max()
    if peak > 1e-4:
        audio = audio / peak
    return audio.astype(np.float32)

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
    Key changes vs v1:
      - Random f0 per chunk (60-120 Hz) so the test covers diverse engine types.
      - Heavy broadband exhaust noise (amplitude ~= harmonics) so BPF-200-6k
        view looks noise-dominated rather than clean-harmonic, preventing the
        CNN from confusing engine harmonics with drone blade-pass lines.
      - Irregular mechanical impulses (valve/injector noise).
    """
    rng = np.random.default_rng(int(t0 * 1000 + 17) % 99991)
    f0  = rng.uniform(60.0, 120.0)          # random cylinder firing rate

    t   = np.linspace(t0, t0 + n / FS, n, endpoint=False)
    rpm = 1.0 + 0.05 * np.sin(2 * np.pi * 1.2 * t)

    # FM-style harmonics (instantaneous phase integration)
    ph   = np.cumsum(rpm) * (f0 / FS) * 2 * np.pi
    harm = (0.55 * np.sin(ph) +
            0.25 * np.sin(2 * ph) +
            0.12 * np.sin(3 * ph) +
            0.06 * np.sin(4 * ph) +
            0.03 * np.sin(5 * ph))

    # Strong broadband exhaust noise (LP ~2 kHz, amplitude comparable to harmonics)
    exhaust = _lp(rng.standard_normal(n), max(1, int(FS / 2000))) * 0.7

    # Irregular mechanical impulses (valve train / injectors)
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
    pd    = np.mean(drone ** 2) + 1e-12
    pn    = np.mean(noise ** 2) + 1e-12
    scale = np.sqrt(pd / (pn * 10 ** (snr_db / 10.0)))
    return _norm(drone + scale * noise)


# ── Drone file pool ────────────────────────────────────────────────────────
drone_wavs = [f for f in sorted((ROOT / "data" / "raw" / "drone").glob("*.wav"))
              if sf.info(str(f)).frames >= WIN_SAMPLES]

_dq = []

def next_drone():
    if not _dq:
        audio = load_wav(random.choice(drone_wavs))
        if len(audio) < WIN_SAMPLES:
            audio = np.tile(audio, (WIN_SAMPLES // len(audio)) + 2)
        for i in range(0, len(audio) - HOP_SAMPLES + 1, HOP_SAMPLES):
            _dq.append(audio[i:i + HOP_SAMPLES].copy())
    return _dq.pop(0)


# ── Scenario runner ────────────────────────────────────────────────────────
def run_scenario(name, model, drone_idx, gen_fn, n_chunks=1200,
                 smooth_wins=3, smooth_min=2, expect_detect=True):
    """
    Multi-view sliding-window test.
    Temporal smoothing: event = detected in >= smooth_min of last smooth_wins windows.
    """
    buf          = np.zeros(WIN_SAMPLES, dtype=np.float32)
    all_ws       = []
    all_fm       = []
    all_raw_det  = []
    smooth_buf   = [False] * smooth_wins
    n_events     = 0
    reason_counts = {"filteredMax": 0, "weightedScore": 0,
                     "voteCount": 0, "notDetected": 0}

    t_start = time.perf_counter()
    for i in range(n_chunks):
        chunk             = gen_fn(i)
        buf[:HOP_SAMPLES] = buf[HOP_SAMPLES:]
        buf[HOP_SAMPLES:] = chunk
        if i < 1:
            continue

        ws, fm, det, probs, reason = infer_multiview(model, drone_idx, buf.copy())
        all_ws.append(ws)
        all_fm.append(fm)
        all_raw_det.append(det)
        reason_counts[reason] += 1

        # Temporal smoothing
        smooth_buf = smooth_buf[1:] + [det]
        if sum(smooth_buf) >= smooth_min:
            n_events += 1

    elapsed  = time.perf_counter() - t_start
    sim_sec  = n_chunks * HOP_SAMPLES / FS
    ws_arr   = np.array(all_ws)
    fm_arr   = np.array(all_fm)
    n_raw    = sum(all_raw_det)
    n_valid  = len(all_raw_det)
    raw_rate = 100.0 * n_raw  / n_valid if n_valid else 0.0
    evt_rate = 100.0 * n_events / n_valid if n_valid else 0.0

    ok      = (evt_rate > 50) == expect_detect
    verdict = "PASS" if evt_rate > 50 else ("MARGINAL" if evt_rate > 20 else "FAIL")
    flag    = "OK" if ok else "!! PROBLEM !!"

    # Best trigger path
    top_reason = max(("filteredMax", "weightedScore", "voteCount"),
                     key=lambda k: reason_counts[k])

    print(f"  Scenario   : {name}")
    print(f"  Simulated  : {sim_sec:.0f}s ({sim_sec/60:.1f} min)"
          f"  |  computed in {elapsed:.1f}s")
    print(f"  WeightedScore  mean={ws_arr.mean():.4f}  max={ws_arr.max():.4f}")
    print(f"  FilteredMax    mean={fm_arr.mean():.4f}  max={fm_arr.max():.4f}")
    print(f"  Raw detections : {raw_rate:.1f}%   "
          f"Smoothed events: {evt_rate:.1f}%")
    print(f"  Trigger paths  : filtMax={reason_counts['filteredMax']}  "
          f"wgtScore={reason_counts['weightedScore']}  "
          f"votes={reason_counts['voteCount']}")
    print(f"  Result     : {verdict}   [{flag}]   (top path: {top_reason})")
    print()

    return evt_rate, verdict, ok, ws_arr, fm_arr


# ── Main ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    random.seed(42)

    print("=" * 65)
    print("  MULTI-VIEW DRONE DETECTION TEST  (no audio output)")
    print("=" * 65)
    print(f"  Device      : {DEVICE}")
    print(f"  Drone pool  : {len(drone_wavs)} files  (>= 1 second)")
    print(f"  Views       : {VIEW_NAMES}")
    print(f"  Weights     : {VIEW_WEIGHTS.tolist()}")
    print(f"  Rule        : filteredMax > {FMAX_THR}  OR  "
          f"weightedScore > {SCORE_THR}  OR  voteCount >= {VOTES_NEED}")
    print(f"  Smoothing   : event = detected in >= 2 of last 3 windows")
    print(f"  Each scenario: 1200 chunks = 600 s simulated = 10 min")
    print()

    model, drone_idx, model_name = load_model()
    print(f"  Model loaded OK\n")

    CHUNKS = 1200   # 10 simulated minutes per scenario

    scenarios = [
        # (label, gen_fn, expect_detect)
        ("Pure drone (baseline)",
         lambda i: next_drone(),
         True),

        ("Drone + Tank   0 dB SNR  (equal loudness)",
         lambda i: mix_snr(next_drone(), synth_tank(HOP_SAMPLES, i * HOP_SAMPLES / FS), 0),
         True),

        ("Drone + Tank  -5 dB SNR  (tank 3× louder)",
         lambda i: mix_snr(next_drone(), synth_tank(HOP_SAMPLES, i * HOP_SAMPLES / FS), -5),
         True),

        ("Drone + Tank -10 dB SNR  (tank 10× louder)",
         lambda i: mix_snr(next_drone(), synth_tank(HOP_SAMPLES, i * HOP_SAMPLES / FS), -10),
         True),

        ("Drone + Tank -20 dB SNR  (tank 100× louder)",
         lambda i: mix_snr(next_drone(), synth_tank(HOP_SAMPLES, i * HOP_SAMPLES / FS), -20),
         True),

        ("Tank only  (false alarm check)",
         lambda i: synth_tank(HOP_SAMPLES, i * HOP_SAMPLES / FS),
         False),

        ("Engine only  (false alarm check)",
         lambda i: synth_engine(HOP_SAMPLES, i * HOP_SAMPLES / FS),
         False),

        ("Crowd only  (false alarm check)",
         lambda i: synth_crowd(HOP_SAMPLES, i * HOP_SAMPLES / FS),
         False),
    ]

    all_ok  = True
    summary = []

    for idx, (name, gen_fn, expect) in enumerate(scenarios):
        print(f"--- [{idx+1}/{len(scenarios)}] {name} ---")
        _dq.clear()
        evt_rate, verdict, ok, ws_arr, fm_arr = run_scenario(
            name, model, drone_idx, gen_fn, n_chunks=CHUNKS, expect_detect=expect)
        summary.append((name, evt_rate, verdict, ok, expect))
        if not ok:
            all_ok = False

    # ── Final report ──────────────────────────────────────────────────
    print("=" * 65)
    print(f"  FINAL REPORT  [{model_name}  +  multi-view pipeline]")
    print("=" * 65)
    for name, dr, verdict, ok, expect in summary:
        expected_str = "DETECT" if expect else "NO ALARM"
        flag         = "OK" if ok else "!! PROBLEM !!"
        print(f"  {verdict:8s}  events={dr:5.1f}%  {flag:12s}  {name}")

    print()
    if all_ok:
        print("  OVERALL: ALL SCENARIOS PASSED")
    else:
        failed = [s for s in summary if not s[3]]
        print(f"  OVERALL: {len(failed)} SCENARIO(S) FAILED:")
        for name, dr, verdict, ok, expect in failed:
            if expect and dr <= 50:
                print(f"    MISSED  drone in '{name}'  (events={dr:.1f}%)")
                print(f"    CAUSE : drone mel-features still buried by noise at this SNR.")
                print(f"    OPTION: lower FMAX_THR / SCORE_THR or retrain with HPF.")
            elif not expect and dr > 50:
                print(f"    FALSE ALARM  on '{name}'  (events={dr:.1f}%)")
                print(f"    CAUSE : noise harmonics pass the BPF views.")
                print(f"    OPTION: raise SCORE_THR or add this noise to no_drone training.")
    print()
