from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

PHASE2V5_BACKBONE_PATH = ROOT / "models" / "phase2v5_real_noise" / "drone_cnn_phase2v5c_real_noise_balanced.pth"
MODELS_DIR = ROOT / "models" / "phase2_harmonic_fusion"
RESULTS_DIR = ROOT / "results" / "phase2_harmonic_fusion"
FEATURES_DIR = RESULTS_DIR / "features"
LOGS_DIR = RESULTS_DIR / "logs"

DEFAULT_SAVE_PATH = MODELS_DIR / "drone_cnn_phase2_harmonic_fusion_v1.pth"

SAMPLE_RATE = 16000
SNR_LEVELS = [-20, -15, -10, -5, 0, 5, 10]

HARMONIC_FEATURE_NAMES = [
    "f0_norm",
    "hps_confidence",
    "low_band_ratio",
    "harmonicity_score",
    "upper_harmonic_explained_ratio",
    "impulse_score",
    "vehicle_risk_score",
    "num_harmonics_norm",
]


def ensure_dirs():
    for path in [MODELS_DIR, RESULTS_DIR, FEATURES_DIR, LOGS_DIR]:
        path.mkdir(parents=True, exist_ok=True)

