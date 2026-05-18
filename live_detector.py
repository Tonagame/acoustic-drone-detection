"""
live_detector.py  --  Real-time drone audio detector with visual alert.

Usage
-----
    python live_detector.py                   # uses VB-Cable (PC audio)
    python live_detector.py --mic             # uses microphone instead
    python live_detector.py --device 1        # pick a specific device index
    python live_detector.py --list-devices    # show all input devices
    python live_detector.py --threshold 0.4   # override starting threshold
    python live_detector.py --test            # TEST MODE (no mic needed)

Audio source
------------
  Default: "CABLE Output (VB-Audio Point)" device 34 — captures PC audio
           directly (YouTube, media player, etc.) with no room noise.
  --mic  : use the default microphone instead.
  --test : bypass microphone entirely; inject synthetic drone / noise audio
           directly into the detector so you can verify it works.
"""

import argparse
import queue
import random
import threading
import time
from pathlib import Path
import tkinter as tk
from tkinter import ttk

import numpy as np
import scipy.signal as ssig
import sounddevice as sd
import soundfile as sf
import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
import torchaudio.functional as FA

# ── Model ─────────────────────────────────────────────────────────────────
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
# Auto-select best available model (v4 > v3 > v3b > phase3 > phase2v2 > ...)
def _best_model():
    priority = [
        'drone_cnn_phase2_v4_specialist_ensemble',
        'drone_cnn_phase2_v3_multiview_hardneg',
        'drone_cnn_phase2_v3b_engine_v2',
        'drone_cnn_phase3',
        'drone_cnn_phase2v2',
        'drone_cnn_phase1b',
        'drone_cnn_phase1',
    ]
    model_dir = ROOT / "models"
    candidates = sorted(model_dir.glob("*.pth")) if model_dir.exists() else []
    for tag in priority:
        for p in candidates:
            if tag in p.stem:
                return p
    return ROOT / "models" / "drone_cnn_phase1.pth"
best_model_path = _best_model()
MODEL_PATH  = best_model_path
USE_ENSEMBLE = 'v4_specialist' in best_model_path.stem
USE_MULTIVIEW = any(tag in MODEL_PATH.name
                    for tag in ("v3", "multiview", "hardneg"))

FS          = 16000
WIN_SAMPLES = FS            # 1-second window
HOP_SAMPLES = FS // 2       # 0.5-second hop

N_FFT   = 512
WIN_LEN = round(0.025 * FS)
HOP_LEN = WIN_LEN - round(0.015 * FS)
N_MELS  = 64

VBCABLE_DEVICE = 34         # "CABLE Output (VB-Audio Point)"
NOISE_FLOOR    = 0.030      # ignore ambient mic noise (measured peak ~0.026 in silence)
SMOOTH_ALPHA   = 0.4        # EMA smoothing (0=no smooth, 1=instant)
HISTORY_LEN    = 30         # number of bars in the history chart

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_mel = T.MelSpectrogram(
    sample_rate=FS, n_fft=N_FFT,
    win_length=WIN_LEN, hop_length=HOP_LEN,
    n_mels=N_MELS, power=2.0,
).to(DEVICE)


# ── Multi-view pipeline (Phase 2v3+) ──────────────────────────────────────
_HP150 = ssig.butter(4, 150,         btype='high', fs=FS, output='sos')
_HP250 = ssig.butter(4, 250,         btype='high', fs=FS, output='sos')
_BP200 = ssig.butter(4, [200, 6000], btype='band', fs=FS, output='sos')
_BP500 = ssig.butter(4, [500, 6000], btype='band', fs=FS, output='sos')

VIEW_WEIGHTS = np.array([0.05, 0.20, 0.25, 0.35, 0.15], dtype=np.float32)
ACTIVE_VIEW_WEIGHTS = VIEW_WEIGHTS.copy()
FMAX_THR     = 0.75   # filteredMax  (views 2-5) threshold
SCORE_THR    = 0.60   # weighted score threshold
VOTE_THR     = 0.60   # per-view threshold for vote counting
VOTES_NEED   = 2      # minimum votes to declare detection

def _fv(s):
    pk = np.abs(s).max()
    return (s / (pk + 1e-9)).astype(np.float32)

def create_audio_views(x: np.ndarray):
    x = x.astype(np.float64)
    return [_fv(x),
            _fv(ssig.sosfiltfilt(_HP150, x)),
            _fv(ssig.sosfiltfilt(_HP250, x)),
            _fv(ssig.sosfiltfilt(_BP200, x)),
            _fv(ssig.sosfiltfilt(_BP500, x))]

def _view_prob(model, drone_idx, view: np.ndarray) -> float:
    peak = np.abs(view).max()
    if peak < NOISE_FLOOR:
        return 0.0
    t   = torch.from_numpy(view).unsqueeze(0).to(DEVICE)
    mel = _mel(t)
    lm  = torch.log10(mel + 1e-10).unsqueeze(0)
    with torch.no_grad():
        return torch.softmax(model(lm), dim=1)[0, drone_idx].item()

def infer_multiview(model, drone_idx, window: np.ndarray):
    """Run all 5 views; return (weighted_score, detected, reason)."""
    # Check amplitude on the ORIGINAL window before any normalisation.
    # (create_audio_views normalises every view to peak=1, so _view_prob's
    # own check would never fire — silence would still score ~30%.)
    if np.abs(window).max() < NOISE_FLOOR:
        return 0.0, False, "silent"

    views = create_audio_views(window)
    probs = np.array([_view_prob(model, drone_idx, v) for v in views],
                     dtype=np.float32)
    ws  = float(VIEW_WEIGHTS @ probs)
    fm  = float(probs[1:].max())          # filtered max (views 2-5)
    vc  = int((probs > VOTE_THR).sum())   # vote count

    path_a = fm  > FMAX_THR
    path_b = ws  > SCORE_THR
    path_c = vc >= VOTES_NEED
    det = path_a or path_b or path_c

    if   path_a: reason = f"filtMax={fm:.2f}"
    elif path_b: reason = f"wgtScore={ws:.2f}"
    elif path_c: reason = f"votes={vc}"
    else:        reason = "none"
    return ws, det, reason


# ג”€ג”€ Specialist ensemble pipeline (Phase 2v4) ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€
def _norm_view_ensemble(x: np.ndarray) -> np.ndarray:
    x = x - x.mean()
    pk = np.abs(x).max()
    return x / pk if pk > 1e-6 else x


def create_audio_views_ensemble(x: np.ndarray):
    """Training-matched v4 views: causal sosfilt, not the legacy live filtfilt."""
    x = x.astype(np.float64)
    x = x - x.mean()
    pk = np.abs(x).max()
    if pk < 1e-6:
        return [np.zeros_like(x, dtype=np.float32)] * 5
    x = x / pk
    return [
        x.astype(np.float32),
        _norm_view_ensemble(ssig.sosfilt(_HP150, x)).astype(np.float32),
        _norm_view_ensemble(ssig.sosfilt(_HP250, x)).astype(np.float32),
        _norm_view_ensemble(ssig.sosfilt(_BP200, x)).astype(np.float32),
        _norm_view_ensemble(ssig.sosfilt(_BP500, x)).astype(np.float32),
    ]


def _ensemble_view_prob(model, drone_idx, view: np.ndarray) -> float:
    peak = np.abs(view).max()
    if peak < NOISE_FLOOR:
        return 0.0
    view = (view / peak).astype(np.float32)
    t   = torch.from_numpy(view).unsqueeze(0).to(DEVICE)
    mel = _mel(t)
    lm  = torch.log10(mel + 1e-10).unsqueeze(0)
    with torch.no_grad():
        return torch.softmax(model(lm), dim=1)[0, drone_idx].item()


@torch.no_grad()
def predict_ensemble(models, audio, drone_idx, device):
    """Run 5 specialist models on their respective views. Return weighted score."""
    views = create_audio_views_ensemble(audio)
    probs = np.zeros(5, dtype=np.float32)
    for vi, (model, view) in enumerate(zip(models, views)):
        probs[vi] = _ensemble_view_prob(model, drone_idx, view)
    return float((ACTIVE_VIEW_WEIGHTS * probs).sum()), probs


def infer_ensemble(models, drone_idx, window: np.ndarray):
    """Run v4 specialist ensemble; return (weighted_score, detected, reason)."""
    if np.abs(window).max() < NOISE_FLOOR:
        return 0.0, False, "silent"

    ws, probs = predict_ensemble(models, window, drone_idx, DEVICE)
    fm  = float(probs[1:].max())
    vc  = int((probs > VOTE_THR).sum())

    path_a = fm > FMAX_THR
    path_b = ws > SCORE_THR
    path_c = vc >= VOTES_NEED
    det = path_a or path_b or path_c

    if   path_a: reason = f"filtMax={fm:.2f}"
    elif path_b: reason = f"wgtScore={ws:.2f}"
    elif path_c: reason = f"votes={vc}"
    else:        reason = "none"
    return ws, det, reason


# ── Model loading ─────────────────────────────────────────────────────────
# These are set by load_model() based on the checkpoint's preprocessing info
_active_mel    = None   # mel transform chosen at load time
_hpf_cutoff_hz = 0      # 0 = no HPF (old models); >0 = apply HPF

def load_model():
    global _active_mel, _hpf_cutoff_hz

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}\nRun train_phase1_gpu.py first.")
    ckpt    = torch.load(str(MODEL_PATH), map_location=DEVICE, weights_only=False)
    classes = ckpt.get("classes", ["drone", "no_drone"])
    model   = DroneCNN(len(classes)).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Read preprocessing params saved during training (phase3b+)
    pre = ckpt.get("preprocessing", {})
    f_min          = pre.get("f_min_mel",     0.0)
    f_max          = pre.get("f_max_mel",  8000.0)
    _hpf_cutoff_hz = int(pre.get("hpf_cutoff_hz", 0))

    _active_mel = T.MelSpectrogram(
        sample_rate=FS, n_fft=N_FFT,
        win_length=WIN_LEN, hop_length=HOP_LEN,
        n_mels=N_MELS, power=2.0,
        f_min=f_min, f_max=f_max,
    ).to(DEVICE)

    phase = ckpt.get("phase", "?")
    print(f"Loaded phase={phase}  HPF={_hpf_cutoff_hz}Hz  "
          f"mel_fmin={f_min}Hz")
    return model, classes.index("drone")


def load_ensemble(path, device):
    global ACTIVE_VIEW_WEIGHTS

    bundle = torch.load(str(path), map_location=device, weights_only=False)
    ACTIVE_VIEW_WEIGHTS = np.array(
        bundle.get("view_weights", VIEW_WEIGHTS.tolist()), dtype=np.float32)
    view_names = bundle.get("view_names", ['raw', 'HPF-150', 'HPF-250', 'BPF-200-6k', 'BPF-500-6k'])
    classes = bundle.get("classes", ["drone", "no_drone"])
    drone_idx = int(bundle.get("drone_idx", classes.index("drone") if "drone" in classes else 0))

    models = []
    for vi, vname in enumerate(view_names):
        key = f'model_{vi}_{vname.replace("-","_").replace("+","_")}'
        if key not in bundle:
            matches = [k for k in bundle if k.startswith(f"model_{vi}_")]
            if not matches:
                raise KeyError(f"Missing specialist state dict for view {vi}: {vname}")
            key = matches[0]
        model = DroneCNN(len(classes)).to(device)
        model.load_state_dict(bundle[key])
        model.eval()
        models.append(model)

    print(f"Loaded phase={bundle.get('phase', 'phase2v4')}  "
          f"specialists={len(models)}  weights={ACTIVE_VIEW_WEIGHTS.tolist()}")
    return models, drone_idx, ACTIVE_VIEW_WEIGHTS


# ── Inference ─────────────────────────────────────────────────────────────
def infer(model, drone_idx, window: np.ndarray) -> float:
    """Return drone probability for a 16000-sample float32 window."""
    peak = np.abs(window).max()
    if peak < NOISE_FLOOR:
        return 0.0
    window = window / peak

    # Apply high-pass filter if the loaded model was trained with one
    if _hpf_cutoff_hz > 0:
        N      = len(window)
        X      = np.fft.rfft(window)
        freqs  = np.fft.rfftfreq(N, d=1.0 / FS)
        X[freqs < _hpf_cutoff_hz] = 0
        window = np.fft.irfft(X, n=N).astype(np.float32)

    mel_fn = _active_mel if _active_mel is not None else _mel
    t      = torch.from_numpy(window).unsqueeze(0).to(DEVICE)
    mel    = mel_fn(t)
    logmel = torch.log10(mel + 1e-10).unsqueeze(0)
    with torch.no_grad():
        prob = torch.softmax(model(logmel), dim=1)[0, drone_idx].item()
    return float(prob)


# ── Synthetic audio generators (test mode) ───────────────────────────────

def _normalize(sig: np.ndarray, level: float = 0.85) -> np.ndarray:
    peak = np.abs(sig).max()
    return (sig / peak * level).astype(np.float32) if peak > 1e-7 else sig.astype(np.float32)

def _lp(sig: np.ndarray, taps: int) -> np.ndarray:
    """Quick moving-average low-pass filter."""
    return np.convolve(sig, np.ones(taps) / taps, mode="same")


def synth_drone(n_samples: int, freq_hz: float, t_offset: float) -> np.ndarray:
    """Drone buzz: fundamental + harmonics + RPM wobble."""
    t  = np.linspace(t_offset, t_offset + n_samples / FS, n_samples, endpoint=False)
    fm = 1.0 + 0.02 * np.sin(2 * np.pi * 3.0 * t)
    s  = (0.50 * np.sin(2*np.pi * freq_hz*1 * fm * t) +
          0.28 * np.sin(2*np.pi * freq_hz*2 * fm * t) +
          0.14 * np.sin(2*np.pi * freq_hz*3 * fm * t) +
          0.08 * np.sin(2*np.pi * freq_hz*4 * fm * t))
    s += 0.05 * np.random.randn(n_samples)
    return _normalize(s)


def synth_random_noise(n_samples: int) -> np.ndarray:
    """Pink-ish band-limited noise."""
    return _normalize(_lp(np.random.randn(n_samples), 8), 0.5)


def synth_tank(n_samples: int, t_offset: float = 0.0) -> np.ndarray:
    """
    Tank / armored vehicle: diesel engine ~45 Hz + harmonics + tracks clank.
    Low, heavy, irregular rumble — should NOT trigger drone detection alone.
    """
    t    = np.linspace(t_offset, t_offset + n_samples / FS, n_samples, endpoint=False)
    # Diesel idle: ~45 Hz with slow RPM drift
    rpm  = 1.0 + 0.04 * np.sin(2*np.pi * 0.3 * t)
    f0   = 45.0
    eng  = (0.55 * np.sin(2*np.pi * f0*1 * rpm * t) +
            0.25 * np.sin(2*np.pi * f0*2 * rpm * t) +
            0.12 * np.sin(2*np.pi * f0*3 * rpm * t) +
            0.08 * np.sin(2*np.pi * f0*4 * rpm * t))
    # Track clank: short random bursts every ~0.15 s
    clank = np.zeros(n_samples)
    rng   = np.random.default_rng(int(t_offset * 100) % 9999)
    for pos in range(0, n_samples, int(FS * 0.15)):
        burst = min(int(FS * 0.01), n_samples - pos)
        clank[pos : pos + burst] = rng.standard_normal(burst) * 0.4
    # Heavy sub-bass rumble
    rumble = _lp(np.random.randn(n_samples), 64) * 0.3
    sig    = eng + clank + rumble
    return _normalize(sig)


def synth_engine(n_samples: int, t_offset: float = 0.0) -> np.ndarray:
    """
    Vehicle engine v2: random f0 (60-120 Hz) + heavy broadband exhaust noise.
    Less likely to false-trigger the multiview CNN than the old clean-harmonic version.
    """
    rng = np.random.default_rng(int(t_offset * 1000 + 17) % 99991)
    f0  = rng.uniform(60.0, 120.0)
    t   = np.linspace(t_offset, t_offset + n_samples / FS, n_samples, endpoint=False)
    rpm = 1.0 + 0.05 * np.sin(2 * np.pi * 1.2 * t)
    ph  = np.cumsum(rpm) * (f0 / FS) * 2 * np.pi
    harm = (0.55 * np.sin(ph) + 0.25 * np.sin(2 * ph) +
            0.12 * np.sin(3 * ph) + 0.06 * np.sin(4 * ph) + 0.03 * np.sin(5 * ph))
    exhaust = _lp(rng.standard_normal(n_samples), max(1, int(FS / 2000))) * 0.7
    mech = np.zeros(n_samples)
    pos  = 0
    while pos < n_samples:
        pos += int(rng.integers(max(1, int(FS * 0.03)), max(2, int(FS * 0.12))))
        if pos >= n_samples:
            break
        b = min(int(rng.integers(1, 6)), n_samples - pos)
        if b > 0:
            mech[pos:pos + b] = rng.standard_normal(b) * rng.uniform(0.05, 0.3)
    return _normalize(harm + exhaust + mech)


def synth_crowd(n_samples: int, t_offset: float = 0.0) -> np.ndarray:
    """
    Crowd / street noise: band-pass 200–3000 Hz with speech-rhythm AM.
    Should NOT trigger drone detection alone (tests speech robustness).
    """
    white  = np.random.randn(n_samples)
    # Band-pass: subtract heavy LP (removes below ~200 Hz)
    hp     = white - _lp(white, 80)
    # Light LP to cut above ~3 kHz
    bp     = _lp(hp, 5)
    # Speech-like amplitude modulation: 3 Hz envelope
    t      = np.linspace(t_offset, t_offset + n_samples / FS, n_samples, endpoint=False)
    am     = 0.4 + 0.6 * np.abs(np.sin(2*np.pi * 3.0 * t))
    return _normalize(bp * am, 0.6)


def mix_at_snr(drone: np.ndarray, noise: np.ndarray, snr_db: float = 0.0) -> np.ndarray:
    """Mix drone and noise signals at the specified SNR (dB)."""
    p_d = np.mean(drone**2) + 1e-12
    p_n = np.mean(noise**2) + 1e-12
    scale = np.sqrt(p_d / (p_n * 10 ** (snr_db / 10.0)))
    return _normalize(drone + scale * noise)


# ── Test signal catalogue ─────────────────────────────────────────────────
# Each entry: (button_label, mode_key, drone_freq_hz, bg_color, row_label)
# row_label is used as the section header
TEST_SIGNALS = [
    # ── Row 1: pure drone (should go RED) ────────────────────────────
    ("DRONE\nreal audio\n(random clip)", "drone", 0, "#c0392b", "drone"),
    # ── Row 2: drone + interference (should STILL go RED) ────────────
    ("DRONE\n+ Tank\n0 dB SNR",  "drone+tank",   220,  "#e67e22", "mixed"),
    ("DRONE\n+ Engine\n0 dB",    "drone+engine", 800,  "#e67e22", "mixed"),
    ("DRONE\n+ Crowd\n0 dB",     "drone+crowd",  220,  "#e67e22", "mixed"),
    # ── Row 3: noise only (should stay GREEN) ────────────────────────
    ("RANDOM\nnoise",             "noise",          0,  "#2980b9", "noise"),
    ("TANK\nonly",                "tank",           0,  "#2980b9", "noise"),
    ("ENGINE\nonly",              "engine",         0,  "#2980b9", "noise"),
    ("CROWD\nonly",               "crowd",          0,  "#2980b9", "noise"),
]


# ── GUI ───────────────────────────────────────────────────────────────────
class DetectorGUI:
    CLR_GREEN  = "#1db954"
    CLR_YELLOW = "#f5a623"
    CLR_RED    = "#e8001c"
    CLR_DARK   = "#1a1a1a"
    CLR_TEST   = "#7b2fff"   # purple accent for test mode

    def __init__(self, root: tk.Tk, threshold: float, device_name: str,
                 test_mode: bool = False):
        self.root       = root
        self._q         = queue.Queue()
        self._smooth    = 0.0
        self._history   = [0.0] * HISTORY_LEN
        self._thr_var   = tk.DoubleVar(value=threshold)
        self.test_mode  = test_mode

        title = "Drone Detector  [TEST MODE]" if test_mode else "Drone Detector"
        root.title(title)
        root.geometry("780x580" if test_mode else "680x480")
        root.configure(bg=self.CLR_DARK)
        root.resizable(True, True)

        # ── Test mode banner ──────────────────────────────────────────
        if test_mode:
            tk.Label(root,
                     text="  TEST MODE — synthetic audio, no microphone  ",
                     font=("Helvetica", 10, "bold"),
                     bg=self.CLR_TEST, fg="white",
                     anchor="center").pack(fill="x")

        # ── Status label ──────────────────────────────────────────────
        self.status_var = tk.StringVar(value="INITIALISING...")
        self.status_lbl = tk.Label(
            root, textvariable=self.status_var,
            font=("Helvetica", 48, "bold"),
            bg=self.CLR_DARK, fg="white",
            justify="center", anchor="center",
        )
        self.status_lbl.pack(expand=True, fill="both", padx=10, pady=(20, 0))

        # ── Probability bar ───────────────────────────────────────────
        self.bar_canvas = tk.Canvas(root, height=36, bg="#333",
                                    highlightthickness=0)
        self.bar_canvas.pack(fill="x", padx=20, pady=6)
        self._bar   = self.bar_canvas.create_rectangle(0, 0, 0, 36,
                                                        fill=self.CLR_GREEN, width=0)
        self._btext = self.bar_canvas.create_text(340, 18, text="0.0%",
                                                   fill="white",
                                                   font=("Helvetica", 13, "bold"))

        # ── History chart ─────────────────────────────────────────────
        self.hist_canvas = tk.Canvas(root, height=70, bg="#222",
                                     highlightthickness=0)
        self.hist_canvas.pack(fill="x", padx=20, pady=2)
        self._hist_bars = []
        for _ in range(HISTORY_LEN):
            r = self.hist_canvas.create_rectangle(0, 0, 0, 0,
                                                   fill="#555", width=0)
            self._hist_bars.append(r)

        # ── Test mode controls ────────────────────────────────────────
        if test_mode:
            self._test_thread_ref = None

            # Section labels + button rows
            ROW_META = {
                "drone": ("DRONE only  (expect RED)",   "#c0392b", "#fff0f0"),
                "mixed": ("DRONE + noise  (expect RED)","#e67e22", "#fff5e0"),
                "noise": ("Noise only  (expect GREEN)", "#2980b9", "#e8f4ff"),
            }
            self._sig_var = tk.StringVar(value="Signal: random noise")

            outer = tk.Frame(root, bg="#1e1e1e")
            outer.pack(fill="x", padx=10, pady=(4, 0))

            rows: dict = {}
            for row_key, (row_lbl, hdr_bg, _) in ROW_META.items():
                frm = tk.Frame(outer, bg="#1e1e1e")
                frm.pack(fill="x", pady=1)
                tk.Label(frm, text=row_lbl, bg="#1e1e1e",
                         fg=hdr_bg, font=("Helvetica", 8, "bold"),
                         width=26, anchor="w").pack(side="left")
                rows[row_key] = frm

            for btn_lbl, mode, freq, bg, row_key in TEST_SIGNALS:
                b = tk.Button(
                    rows[row_key],
                    text=btn_lbl,
                    font=("Helvetica", 7, "bold"),
                    bg=bg, fg="white",
                    activebackground=bg, activeforeground="white",
                    relief="flat", padx=5, pady=2,
                    command=lambda m=mode, f=freq, l=btn_lbl.replace("\n", " "):
                        self._set_test(m, f, l)
                )
                b.pack(side="left", padx=2)

            # Current signal readout
            tk.Label(outer, textvariable=self._sig_var,
                     bg="#1e1e1e", fg=self.CLR_TEST,
                     font=("Helvetica", 9, "bold"),
                     anchor="w").pack(fill="x", pady=(2, 0))

        # ── Threshold slider ──────────────────────────────────────────
        ctrl = tk.Frame(root, bg=self.CLR_DARK)
        ctrl.pack(fill="x", padx=20, pady=4)
        tk.Label(ctrl, text="Sensitivity:", bg=self.CLR_DARK, fg="#aaa",
                 font=("Helvetica", 10)).pack(side="left")
        self._thr_lbl = tk.Label(ctrl, text=f"{threshold:.2f}",
                                  bg=self.CLR_DARK, fg="white",
                                  font=("Helvetica", 10, "bold"), width=4)
        self._thr_lbl.pack(side="right")
        ttk.Scale(ctrl, from_=0.05, to=0.95, variable=self._thr_var,
                  orient="horizontal",
                  command=lambda v: self._thr_lbl.config(
                      text=f"{float(v):.2f}")
                  ).pack(side="left", fill="x", expand=True, padx=8)

        # ── Info bar ──────────────────────────────────────────────────
        src      = "SYNTHETIC" if test_mode else device_name
        pipeline = "specialist-5" if USE_ENSEMBLE else (
            "multi-view" if USE_MULTIVIEW else "single-view")
        tk.Label(root,
                 text=f"Source: {src}   |   {MODEL_PATH.name}   |   "
                      f"{pipeline}   |   {DEVICE}",
                 font=("Helvetica", 9), bg="#111", fg="#666",
                 anchor="center").pack(fill="x")

        root.after(50, self._poll)

    def _set_test(self, mode: str, freq: float, label: str):
        if self._test_thread_ref is not None:
            self._test_thread_ref.set_signal(mode, freq)
        self._sig_var.set(f"Signal: {label}")

    @property
    def threshold(self) -> float:
        return self._thr_var.get()

    def push(self, prob: float):
        self._q.put(prob)

    def _poll(self):
        try:
            while True:
                raw = self._q.get_nowait()
                self._smooth = SMOOTH_ALPHA * raw + (1 - SMOOTH_ALPHA) * self._smooth
                self._history.append(self._smooth)
                self._history.pop(0)
        except queue.Empty:
            pass
        self._refresh()
        self.root.after(50, self._poll)

    def _refresh(self):
        p   = self._smooth
        thr = self.threshold

        if p >= thr:
            bg, fg, colour = self.CLR_RED,    "white", self.CLR_RED
            label = f"DRONE DETECTED\n{p*100:.1f}%"
        elif p >= thr * 0.6:
            bg, fg, colour = self.CLR_YELLOW, "#111", self.CLR_YELLOW
            label = f"POSSIBLE DRONE\n{p*100:.1f}%"
        else:
            bg, fg, colour = self.CLR_GREEN,  "white", self.CLR_GREEN
            label = f"CLEAR\n{p*100:.1f}%"

        self.root.configure(bg=bg)
        self.status_lbl.configure(bg=bg, fg=fg)
        self.status_var.set(label)

        w = max(self.bar_canvas.winfo_width(), 1)
        self.bar_canvas.coords(self._bar, 0, 0, int(w * p), 36)
        self.bar_canvas.itemconfig(self._bar, fill=colour)
        self.bar_canvas.itemconfig(self._btext, text=f"{p*100:.1f}%")
        self.bar_canvas.coords(self._btext, w // 2, 18)

        hw = max(self.hist_canvas.winfo_width(), 1)
        hh = 70
        bw = hw / HISTORY_LEN
        for i, val in enumerate(self._history):
            x0    = i * bw + 1
            x1    = x0 + bw - 2
            bar_h = int(val * hh)
            y0    = hh - bar_h
            clr   = self.CLR_RED if val >= thr else (
                    self.CLR_YELLOW if val >= thr * 0.6 else "#555")
            self.hist_canvas.coords(self._hist_bars[i], x0, y0, x1, hh)
            self.hist_canvas.itemconfig(self._hist_bars[i], fill=clr)

        ty = hh - int(thr * hh)
        self.hist_canvas.delete("thr_line")
        self.hist_canvas.create_line(0, ty, hw, ty, fill="white",
                                      dash=(4, 4), tags="thr_line")


# ── Live inference thread (real microphone / VB-Cable) ────────────────────
class InferenceThread(threading.Thread):
    def __init__(self, gui: DetectorGUI, audio_device):
        super().__init__(daemon=True)
        self.gui  = gui
        self.dev  = audio_device
        self._buf = np.zeros(WIN_SAMPLES, dtype=np.float32)
        self._aq  = queue.Queue()

    def _cb(self, indata, frames, t, status):
        self._aq.put(indata[:, 0].copy())

    def run(self):
        if USE_ENSEMBLE:
            model, drone_idx, _ = load_ensemble(MODEL_PATH, DEVICE)
        else:
            model, drone_idx = load_model()
        print(f"Model ready on {DEVICE}.")

        with sd.InputStream(samplerate=FS, channels=1, dtype="float32",
                            blocksize=HOP_SAMPLES, device=self.dev,
                            callback=self._cb):
            chunks_seen = 0
            while True:
                chunk = self._aq.get()
                self._buf[:HOP_SAMPLES] = self._buf[HOP_SAMPLES:]
                self._buf[HOP_SAMPLES:] = chunk
                chunks_seen += 1
                if chunks_seen < 2:
                    continue
                if USE_ENSEMBLE:
                    prob, _, _ = infer_ensemble(model, drone_idx, self._buf.copy())
                elif USE_MULTIVIEW:
                    prob, _, _ = infer_multiview(model, drone_idx, self._buf.copy())
                else:
                    prob = infer(model, drone_idx, self._buf.copy())
                self.gui.push(prob)


# ── Test inference thread (synthetic audio) ───────────────────────────────
class TestInferenceThread(threading.Thread):
    """
    Plays REAL drone WAV files from data/raw/drone/ through speakers and
    feeds the same audio to the model.  For noise-only tests it uses the
    real no_drone WAV files; for mixed tests it mixes drone WAV + synthetic
    background at 0 dB SNR.
    """
    def __init__(self, gui: DetectorGUI):
        super().__init__(daemon=True)
        self.gui    = gui
        self._buf   = np.zeros(WIN_SAMPLES, dtype=np.float32)
        self._mode  = "noise"
        self._freq  = 0.0    # unused for WAV-based modes
        self._t     = 0.0
        self._lock  = threading.Lock()

        # Index real WAV files
        drone_dir  = ROOT / "data" / "raw" / "drone"
        noise_dir  = ROOT / "data" / "raw" / "no_drone"

        # Only use drone files >= 1 second: these were used in training,
        # and the model detects ~80% of them.  The 0.5-second files were
        # skipped during feature extraction and the model barely recognises
        # them when tiled, giving false "no drone" results in test mode.
        all_drone = sorted(drone_dir.glob("*.wav"))
        self._drone_wavs = [f for f in all_drone
                            if sf.info(str(f)).frames >= WIN_SAMPLES]
        if not self._drone_wavs:          # fallback if dir is empty / unusual
            self._drone_wavs = all_drone

        self._noise_wavs  = sorted(noise_dir.glob("*.wav"))
        print(f"[TEST] {len(self._drone_wavs):,} long drone WAVs (>=1 s) "
              f"of {len(all_drone):,} total  "
              f"| {len(self._noise_wavs):,} no-drone WAVs")

        # Chunk queues — refilled from random WAV files
        self._drone_q : list = []
        self._noise_q : list = []

    def set_signal(self, mode: str, freq: float):
        with self._lock:
            self._mode = mode
            self._freq = freq
            self._t    = 0.0
            self._drone_q.clear()   # force fresh file on next chunk

    # ── WAV helpers ───────────────────────────────────────────────────────
    def _load_wav(self, path: Path) -> np.ndarray:
        """Load, mono, resample to FS, normalize → float32."""
        try:
            audio, sr = sf.read(str(path), dtype="float32")
        except Exception:
            return np.zeros(WIN_SAMPLES, dtype=np.float32)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != FS:
            t     = torch.from_numpy(audio).unsqueeze(0)
            audio = FA.resample(t, sr, FS).squeeze(0).numpy()
        peak = np.abs(audio).max()
        if peak > 1e-4:
            audio = audio / peak * 0.85
        return audio.astype(np.float32)

    def _refill(self, queue: list, wav_list: list):
        """Load a random WAV and push its HOP_SAMPLES chunks onto queue."""
        if not wav_list:
            queue.append(synth_random_noise(HOP_SAMPLES))
            return
        audio = self._load_wav(random.choice(wav_list))
        # If shorter than one window, tile it
        if len(audio) < WIN_SAMPLES:
            reps  = (WIN_SAMPLES // len(audio)) + 2
            audio = np.tile(audio, reps)
        for i in range(0, len(audio) - HOP_SAMPLES + 1, HOP_SAMPLES):
            queue.append(audio[i : i + HOP_SAMPLES].copy())

    def _next_drone(self) -> np.ndarray:
        if not self._drone_q:
            self._refill(self._drone_q, self._drone_wavs)
        return self._drone_q.pop(0)

    def _next_noise(self) -> np.ndarray:
        if not self._noise_q:
            self._refill(self._noise_q, self._noise_wavs)
        return self._noise_q.pop(0)

    # ── Chunk generator ───────────────────────────────────────────────────
    def _generate(self, mode: str, t_off: float) -> np.ndarray:
        if mode == "drone":
            return self._next_drone()
        elif mode == "noise":
            return self._next_noise()
        elif mode == "tank":
            return synth_tank(HOP_SAMPLES, t_off)
        elif mode == "engine":
            return synth_engine(HOP_SAMPLES, t_off)
        elif mode == "crowd":
            return synth_crowd(HOP_SAMPLES, t_off)
        elif mode == "drone+tank":
            return mix_at_snr(self._next_drone(),
                              synth_tank(HOP_SAMPLES, t_off),   snr_db=0)
        elif mode == "drone+engine":
            return mix_at_snr(self._next_drone(),
                              synth_engine(HOP_SAMPLES, t_off), snr_db=0)
        elif mode == "drone+crowd":
            return mix_at_snr(self._next_drone(),
                              synth_crowd(HOP_SAMPLES, t_off),  snr_db=0)
        else:
            return synth_random_noise(HOP_SAMPLES)

    # ── Main loop ─────────────────────────────────────────────────────────
    def run(self):
        import traceback as _tb

        # ── Load model ────────────────────────────────────────────────
        try:
            if USE_ENSEMBLE:
                model, drone_idx, _ = load_ensemble(MODEL_PATH, DEVICE)
            else:
                model, drone_idx = load_model()
            print(f"[TEST] Model loaded OK on {DEVICE}")
        except Exception as e:
            print(f"[TEST] FAILED to load model: {e}")
            _tb.print_exc()
            return

        # ── Open one continuous audio output stream ───────────────────
        # sd.OutputStream.write(chunk) blocks until the hardware
        # consumes the chunk, giving us gapless, stutter-free playback
        # and natural real-time pacing — no sleep() needed.
        stream    = None
        audio_ok  = False
        try:
            stream = sd.OutputStream(samplerate=FS, channels=1,
                                     dtype="float32",
                                     blocksize=HOP_SAMPLES)
            stream.start()
            audio_ok = True
            print("[TEST] Audio stream opened OK")
        except Exception as ae:
            print(f"[TEST] Audio stream failed (detection still runs): {ae}")

        n = 0
        print("[TEST] Inference loop starting...")

        while True:
            try:
                with self._lock:
                    mode  = self._mode
                    t_off = self._t

                chunk = self._generate(mode, t_off)

                with self._lock:
                    self._t += HOP_SAMPLES / FS

                # ── Write to continuous stream (blocks ~0.5 s) ────────
                if audio_ok and stream is not None:
                    try:
                        stream.write(chunk.reshape(-1, 1))
                    except Exception as ae:
                        print(f"[TEST] Audio write error: {ae}")
                        audio_ok = False
                else:
                    # No audio: pace manually
                    time.sleep(HOP_SAMPLES / FS)

                # ── Feed to model ─────────────────────────────────────
                self._buf[:HOP_SAMPLES] = self._buf[HOP_SAMPLES:]
                self._buf[HOP_SAMPLES:] = chunk
                n += 1

                if n >= 2:                        # need 2 chunks to fill 1-sec window
                    if USE_ENSEMBLE:
                        prob, det, reason = infer_ensemble(model, drone_idx,
                                                           self._buf.copy())
                    elif USE_MULTIVIEW:
                        prob, det, reason = infer_multiview(model, drone_idx,
                                                            self._buf.copy())
                    else:
                        prob   = infer(model, drone_idx, self._buf.copy())
                        det    = prob > 0.5
                        reason = ""
                    self.gui.push(prob)
                    if n % 10 == 0:               # print every ~5 seconds
                        tag = "DRONE" if det else "clear"
                        print(f"[TEST] chunk={n:4d}  mode={mode:14s}"
                              f"  score={prob:.4f}  {tag}"
                              + (f"  [{reason}]" if reason else ""))

            except Exception as e:
                print(f"[TEST] Loop error: {e}")
                _tb.print_exc()
                time.sleep(0.1)   # brief pause so we don't spin on error


# ── Entry point ───────────────────────────────────────────────────────────
def pick_device(args) -> tuple:
    if args.device is not None:
        name = sd.query_devices(args.device)["name"]
        return args.device, name

    if args.mic:
        d = sd.default.device[0]
        return d, sd.query_devices(d)["name"]

    devs = sd.query_devices()
    for i, d in enumerate(devs):
        if "CABLE Output" in d["name"] and d["max_input_channels"] > 0:
            print(f"Using VB-Cable loopback: [{i}] {d['name']}")
            print("  -> Make sure 'CABLE Input' is set as your Windows playback device,")
            print("     OR use: python live_detector.py --mic  to use the microphone.\n")
            return i, d["name"]

    d = sd.default.device[0]
    return d, sd.query_devices(d)["name"]


def main():
    parser = argparse.ArgumentParser(description="Live drone detector")
    parser.add_argument("--threshold",    type=float, default=0.60)
    parser.add_argument("--device",       type=int,   default=None)
    parser.add_argument("--mic",          action="store_true",
                        help="Use default microphone instead of VB-Cable")
    parser.add_argument("--test",         action="store_true",
                        help="Test mode: inject synthetic drone/noise audio "
                             "(no microphone needed)")
    parser.add_argument("--list-devices", action="store_true")
    args = parser.parse_args()

    if args.list_devices:
        print(sd.query_devices())
        return

    root = tk.Tk()

    if args.test:
        gui    = DetectorGUI(root, threshold=args.threshold,
                             device_name="SYNTHETIC", test_mode=True)
        thread = TestInferenceThread(gui)
        gui._test_thread_ref = thread   # wire up button callbacks
    else:
        dev_idx, dev_name = pick_device(args)
        gui    = DetectorGUI(root, threshold=args.threshold,
                             device_name=dev_name, test_mode=False)
        thread = InferenceThread(gui, audio_device=dev_idx)

    thread.start()
    root.mainloop()


if __name__ == "__main__":
    main()
