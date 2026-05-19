from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]

DATA_DIR = ROOT / "data"
DRONE_DIR = DATA_DIR / "raw" / "drone"
NODRONE_DIR = DATA_DIR / "raw" / "no_drone"
FSD50K_DIR = DATA_DIR / "external" / "FSD50K"
FSD50K_CANDIDATES_CSV = FSD50K_DIR / "fsd50k_vehicle_engine_candidates.csv"

MODELS_DIR = ROOT / "models" / "phase2v5_real_noise"
RESULTS_DIR = ROOT / "results" / "phase2v5_real_noise"
LATENTS_DIR = RESULTS_DIR / "latents"
LOGS_DIR = RESULTS_DIR / "logs"

DEFAULT_SAVE_PATH = MODELS_DIR / "drone_cnn_phase2v5_real_noise_generalist.pth"
QUICK_SAVE_PATH = MODELS_DIR / "drone_cnn_phase2v5_real_noise_generalist_quick.pth"

FS = 16000
WIN_SAMPLES = 16000
HOP_SAMPLES = 8000
NOISE_FLOOR = 0.002

VIEW_NAMES = ["raw", "HPF-150", "HPF-250", "BPF-200-6k", "BPF-500-6k"]
VIEW_WEIGHTS = np.array([0.05, 0.20, 0.25, 0.35, 0.15], dtype=np.float32)

SNR_LEVELS = [-20, -15, -10, -5, 0, 5, 10]

FSD_LABELS = [
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


def ensure_dirs():
    for path in [MODELS_DIR, RESULTS_DIR, LATENTS_DIR, LOGS_DIR]:
        path.mkdir(parents=True, exist_ok=True)
