"""Microphone-array geometry loading and validation."""

import csv
import json
from pathlib import Path

import numpy as np


def _demo_square_4(spacing_m: float) -> np.ndarray:
    s = float(spacing_m) / 2.0
    return np.array([
        [ s,  s, 0.0],
        [ s, -s, 0.0],
        [-s,  s, 0.0],
        [-s, -s, 0.0],
    ], dtype=np.float64)


def _demo_cube_8(spacing_m: float) -> np.ndarray:
    s = float(spacing_m) / 2.0
    pts = []
    for x in (-s, s):
        for y in (-s, s):
            for z in (-s, s):
                pts.append([x, y, z])
    return np.array(pts, dtype=np.float64)


def _load_json(path: Path) -> np.ndarray:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("mic_positions", data.get("positions", data))
    return np.asarray(data, dtype=np.float64)


def _load_csv(path: Path) -> np.ndarray:
    rows = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames and {"x", "y", "z"}.issubset(set(reader.fieldnames)):
            for row in reader:
                rows.append([float(row["x"]), float(row["y"]), float(row["z"])])
        else:
            f.seek(0)
            plain = csv.reader(f)
            for row in plain:
                if len(row) >= 3:
                    rows.append([float(row[0]), float(row[1]), float(row[2])])
    return np.asarray(rows, dtype=np.float64)


def validate_mic_positions(mic_positions: np.ndarray, expected_channels: int | None = None) -> np.ndarray:
    mic_positions = np.asarray(mic_positions, dtype=np.float64)
    if mic_positions.ndim != 2 or mic_positions.shape[1] != 3:
        raise ValueError(f"Mic positions must have shape [M, 3], got {mic_positions.shape}")
    if mic_positions.shape[0] < 2:
        raise ValueError("At least two microphones are required for beamforming.")
    if not np.isfinite(mic_positions).all():
        raise ValueError("Mic positions contain non-finite values.")
    if expected_channels is not None and mic_positions.shape[0] != expected_channels:
        raise ValueError(
            f"Mic geometry has {mic_positions.shape[0]} mics, but WAV has {expected_channels} channels."
        )
    return mic_positions


def load_array_geometry(config, expected_channels: int | None = None) -> np.ndarray:
    mode = getattr(config, "geometry_mode", "demo_cube_8")
    spacing = getattr(config, "mic_spacing_m", 0.05)

    if mode == "demo_square_4":
        positions = _demo_square_4(spacing)
    elif mode == "demo_cube_8":
        positions = _demo_cube_8(spacing)
    elif mode == "custom_file":
        json_path = Path(getattr(config, "custom_geometry_json"))
        csv_path = Path(getattr(config, "custom_geometry_csv"))
        if json_path.exists():
            positions = _load_json(json_path)
        elif csv_path.exists():
            positions = _load_csv(csv_path)
        else:
            raise FileNotFoundError(
                f"No custom mic geometry found at {json_path} or {csv_path}"
            )
    else:
        raise ValueError(f"Unknown geometry_mode: {mode}")

    return validate_mic_positions(positions, expected_channels)
