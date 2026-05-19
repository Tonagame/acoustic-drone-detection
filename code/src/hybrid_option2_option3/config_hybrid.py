"""
Configuration for the experimental Option2 + Option3 hybrid detector.

This file is intentionally separate from the existing project scripts. Editing
these paths or thresholds does not promote the hybrid or modify any trained
model.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

SRC_DIR = ROOT / "src" / "hybrid_option2_option3"
MODELS_DIR = ROOT / "models"
HYBRID_MODELS_DIR = ROOT / "models" / "hybrid_option2_option3"
RESULTS_DIR = ROOT / "results" / "hybrid_option2_option3"
TIMELINE_DIR = RESULTS_DIR / "hybrid_timeline_plots"
CONFUSION_DIR = RESULTS_DIR / "confusion_matrices"
LOGS_DIR = RESULTS_DIR / "logs"

DATA_DIR = ROOT / "data"
DRONE_DIR = DATA_DIR / "raw" / "drone"
NODRONE_DIR = DATA_DIR / "raw" / "no_drone"
NOISE_DIR = DATA_DIR / "noise"
SPEECH_DIRS = [
    DATA_DIR / "raw" / "speech",
    NOISE_DIR / "speech",
]
WIND_DIRS = [
    NOISE_DIR / "wind",
]

OPTION2_MODEL_PATH = MODELS_DIR / "drone_cnn_phase2_v3_multiview_hardnegatives.pth"
OPTION3_MODEL_PATH = MODELS_DIR / "drone_cnn_phase2_v4_specialist_ensemble.pth"

FS = 16000
WIN_SAMPLES = 16000
HOP_SAMPLES = 8000
NOISE_FLOOR = 0.002

VIEW_NAMES = ["raw", "hpf150", "hpf250", "bpf200_6000", "bpf500_6000"]
VIEW_WEIGHTS = [0.05, 0.20, 0.25, 0.35, 0.15]

OPTION3_SCORE_METHOD = "weighted_average"  # weighted_average, filtered_max, voting
OPTION3_VOTE_THRESHOLD = 0.60
OPTION3_VOTES_NEED = 2

OPTION3_ALONE_FMAX_THR = 0.75
OPTION3_ALONE_SCORE_THR = 0.60
OPTION3_ALONE_VOTE_THR = 0.60
OPTION3_ALONE_VOTES_NEED = 2

HYBRID_RULE = "B"  # A=sensitive, B=balanced, C=conservative, D=two_stage
ENABLE_TANK_ENGINE_VETO = True
VETO_OPTION2_MAX = 0.25
VETO_MAINLY_ONE_VIEW_COUNT = 1

SMOOTHING_MODE = "2of3"  # none, 2of3, 3of5, persist_1_5s

BATCH_SIZE = 64
DEFAULT_N_WINDOWS = 600
MAX_DRONE_FILES = 1500
RANDOM_SEED = 42

OPTION3_THRESHOLD_SWEEP = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
OPTION2_THRESHOLD_SWEEP = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]


def ensure_dirs():
    for d in (HYBRID_MODELS_DIR, RESULTS_DIR, TIMELINE_DIR, CONFUSION_DIR, LOGS_DIR):
        d.mkdir(parents=True, exist_ok=True)
