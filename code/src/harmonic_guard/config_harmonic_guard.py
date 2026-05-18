"""Configuration for the experimental tank/engine/generator harmonic guard.

This iteration is guard-only: it does not modify audio, train CNNs, overwrite
models, or promote itself as the primary detector.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

SRC_DIR = ROOT / "src" / "harmonic_guard"
HYBRID_SRC_DIR = ROOT / "src" / "hybrid_option2_option3"

RESULTS_DIR = ROOT / "results" / "harmonic_guard"
LOGS_DIR = RESULTS_DIR / "logs"

FS = 16000
WIN_SAMPLES = 16000

LOW_F0_MIN_HZ = 30.0
LOW_F0_MAX_HZ = 150.0
HARMONIC_MAX_HZ = 4000.0

HPS_HARMONICS = 5
HARMONIC_TOLERANCE_HZ = 8.0

VEHICLE_RISK_THRESHOLD = 0.68
VEHICLE_RISK_STRONG_THRESHOLD = 0.80
OPTION2_WEAK_MAX = 0.50
OPTION2_STRONG_MIN = 0.65
OPTION3_MEDIUM_MAX = 0.92
HOT_VIEW_THRESHOLD = 0.60
HOT_VIEWS_SPARSE_MAX = 1
VOTE_COUNT_CONFIRM_MIN = 2

DEFAULT_N_WINDOWS = 120
RANDOM_SEED = 4242


def ensure_dirs():
    for d in (RESULTS_DIR, LOGS_DIR):
        d.mkdir(parents=True, exist_ok=True)
