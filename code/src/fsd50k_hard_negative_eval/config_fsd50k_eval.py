"""Configuration for the FSD50K real negative benchmark."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

FSD50K_DIR = ROOT / "data" / "external" / "FSD50K"
FSD50K_CANDIDATES_CSV = FSD50K_DIR / "fsd50k_vehicle_engine_candidates.csv"

HYBRID_SRC_DIR = ROOT / "src" / "hybrid_option2_option3"
HARMONIC_GUARD_SRC_DIR = ROOT / "src" / "harmonic_guard"

RESULTS_DIR = ROOT / "results" / "fsd50k_hard_negative_eval"
WORST_DIR = RESULTS_DIR / "worst_false_alarms"
LOGS_DIR = RESULTS_DIR / "logs"

FS = 16000
WIN_SAMPLES = 16000
HOP_SAMPLES = 8000

LABELS = [
    "Engine",
    "Engine_starting",
    "Motor_vehicle_(road)",
    "Vehicle",
    "Truck",
    "Car",
    "Car_passing_by",
    "Bus",
    "Motorcycle",
    "Aircraft",
    "Explosion",
    "Gunshot_and_gunfire",
]

DEFAULT_MAX_CLIPS_PER_LABEL = 100
DEFAULT_MAX_WINDOWS_PER_CLIP = 4
DEFAULT_MIXED_DRONE_WINDOWS_PER_LABEL = 120
MIXED_SNR_LEVELS = [0, -5, -10]
RANDOM_SEED = 5150


def ensure_dirs():
    for d in (RESULTS_DIR, WORST_DIR, LOGS_DIR):
        d.mkdir(parents=True, exist_ok=True)
