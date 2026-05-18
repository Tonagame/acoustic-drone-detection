from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

MODELS_DIR = ROOT / "models" / "phase2b_pitch_guard"
RESULTS_DIR = ROOT / "results" / "phase2b_pitch_guard"
FEATURES_DIR = RESULTS_DIR / "features"
LOGS_DIR = RESULTS_DIR / "logs"

PHASE3_SPECIALIST_PATH = ROOT / "models" / "phase3_real_noise_specialists" / "drone_cnn_phase3_real_noise_specialist_ensemble.pth"
PHASE2_GUARD_PATH = ROOT / "models" / "phase2_harmonic_fusion" / "drone_cnn_phase2_harmonic_fusion_v1.pth"
PHASE2_BACKBONE_PATH = ROOT / "models" / "phase2v5_real_noise" / "drone_cnn_phase2v5c_real_noise_balanced.pth"

SAVE_PATH = MODELS_DIR / "drone_cnn_phase2b_learned_pitch_guard_v1.pth"

SNR_LEVELS = [-20, -15, -10, -5, 0, 5, 10]
SAMPLE_RATE = 16000
CREPE_FMIN = 50.0
CREPE_FMAX = 550.0
CREPE_STEP_MS = 10
CREPE_MODEL = "tiny"

FEATURE_NAMES = [
    "p_raw",
    "p_hpf150",
    "p_hpf250",
    "p_bpf200",
    "p_bpf500",
    "specialist_score",
    "specialist_filtered_max",
    "specialist_vote_norm",
    "phase2_score",
    "vehicle_risk_score",
    "f0_norm",
    "harmonicity_score",
    "crepe_pitch_median_norm",
    "crepe_pitch_iqr_norm",
    "crepe_periodicity_mean",
    "crepe_periodicity_max",
    "crepe_voiced_ratio",
    "crepe_low_pitch_ratio",
    "crepe_pitch_stability",
]


def ensure_dirs():
    for path in [MODELS_DIR, RESULTS_DIR, FEATURES_DIR, LOGS_DIR]:
        path.mkdir(parents=True, exist_ok=True)

