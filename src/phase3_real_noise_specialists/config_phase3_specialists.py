from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

MODELS_DIR = ROOT / "models" / "phase3_real_noise_specialists"
RESULTS_DIR = ROOT / "results" / "phase3_real_noise_specialists"
CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"
LOGS_DIR = RESULTS_DIR / "logs"

SPECIALIST_BUNDLE_PATH = MODELS_DIR / "drone_cnn_phase3_real_noise_specialist_ensemble.pth"
PHASE2_GUARD_PATH = ROOT / "models" / "phase2_harmonic_fusion" / "drone_cnn_phase2_harmonic_fusion_v1.pth"
PHASE2_BACKBONE_PATH = ROOT / "models" / "phase2v5_real_noise" / "drone_cnn_phase2v5c_real_noise_balanced.pth"

FS = 16000
VIEW_NAMES = ["raw", "HPF-150", "HPF-250", "BPF-200-6k", "BPF-500-6k"]
VIEW_WEIGHTS = np.array([0.05, 0.20, 0.25, 0.35, 0.15], dtype=np.float32)
SNR_LEVELS = [-20, -15, -10, -5, 0, 5, 10]

SPECIALIST_FMAX_THR = 0.78
SPECIALIST_SCORE_THR = 0.60
SPECIALIST_VOTE_THR = 0.60
SPECIALIST_VOTES_NEED = 2

PHASE2_CONFIRM_THR = 0.65
PHASE2_STRONG_THR = 0.85
SPECIALIST_STRONG_THR = 0.90
VEHICLE_RISK_VETO_THR = 0.72
SPARSE_HOT_VIEW_MAX = 1
TEMPORAL_SMOOTHING = "2_of_3"


def ensure_dirs():
    for path in [MODELS_DIR, RESULTS_DIR, CHECKPOINT_DIR, LOGS_DIR]:
        path.mkdir(parents=True, exist_ok=True)

