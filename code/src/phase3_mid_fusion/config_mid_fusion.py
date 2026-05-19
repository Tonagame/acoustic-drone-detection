from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

MODELS_DIR = ROOT / "models" / "phase3_mid_fusion"
RESULTS_DIR = ROOT / "results" / "phase3_mid_fusion"
FEATURES_DIR = RESULTS_DIR / "features"
LOGS_DIR = RESULTS_DIR / "logs"

SPECIALIST_BUNDLE_PATH = ROOT / "models" / "phase3_real_noise_specialists" / "drone_cnn_phase3_real_noise_specialist_ensemble.pth"
PHASE2_GUARD_PATH = ROOT / "models" / "phase2_harmonic_fusion" / "drone_cnn_phase2_harmonic_fusion_v1.pth"
PHASE2_BACKBONE_PATH = ROOT / "models" / "phase2v5_real_noise" / "drone_cnn_phase2v5c_real_noise_balanced.pth"
SAVE_PATH = MODELS_DIR / "drone_cnn_phase3_mid_fusion_v1.pth"
GUARD_NECK_SAVE_PATH = MODELS_DIR / "drone_cnn_phase3_guard_neck_fusion_v2.pth"

FS = 16000
VIEW_NAMES = ["raw", "HPF-150", "HPF-250", "BPF-200-6k", "BPF-500-6k"]
SNR_LEVELS = [-20, -15, -10, -5, 0, 5, 10]
LATENT_DIM_PER_VIEW = 64
MID_FUSION_INPUT_DIM = LATENT_DIM_PER_VIEW * len(VIEW_NAMES)
GUARD_FEATURE_NAMES = ["phase2_guard_score", "vehicle_risk_score", "f0_norm", "harmonicity_score"]
GUARD_NECK_INPUT_DIM = MID_FUSION_INPUT_DIM + len(GUARD_FEATURE_NAMES)
TEMPORAL_SMOOTHING = "2_of_3"


def ensure_dirs():
    for path in [MODELS_DIR, RESULTS_DIR, FEATURES_DIR, LOGS_DIR]:
        path.mkdir(parents=True, exist_ok=True)
