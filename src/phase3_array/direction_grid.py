"""Direction-grid helpers for passive beam scanning."""

import numpy as np


def unit_vector_from_az_el(az_deg: float, el_deg: float) -> np.ndarray:
    az = np.deg2rad(float(az_deg))
    el = np.deg2rad(float(el_deg))
    return np.array([
        np.cos(el) * np.cos(az),
        np.cos(el) * np.sin(az),
        np.sin(el),
    ], dtype=np.float64)


def make_direction_grid(azimuths_deg, elevations_deg) -> list[dict]:
    directions = []
    for el in elevations_deg:
        for az in azimuths_deg:
            directions.append({
                "az_deg": float(az),
                "el_deg": float(el),
                "unit_vector": unit_vector_from_az_el(az, el),
            })
    return directions
