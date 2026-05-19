"""
Phase 3 passive microphone-array configuration.

This phase performs inference/evaluation only. It does not train CNNs and does
not promote or overwrite any existing detector.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

sample_rate_target = 16000
speed_of_sound = 343.0
window_sec = 1.0
hop_sec = 0.5

azimuth_grid_deg = [0, 45, 90, 135, 180, 225, 270, 315]
elevation_grid_deg = [20, 40, 60, 80]

optional_highpass_hz = 100.0  # Set to None to disable.
beam_score_mode = "phase2_harmonic"
detector_mode = "phase2_harmonic"  # phase2_harmonic or hybrid_option2_option3
max_directions_for_full_hybrid = 32
use_energy_prefilter = True
top_k_beams_for_hybrid = 5
smoothing_mode = "2_of_3"  # none, 2_of_3, 3_of_5, persist_1_5s

geometry_mode = "demo_cube_8"  # demo_square_4, demo_cube_8, custom_file
mic_spacing_m = 0.05
custom_geometry_json = ROOT / "data" / "array_geometry" / "mic_positions.json"
custom_geometry_csv = ROOT / "data" / "array_geometry" / "mic_positions.csv"

array_raw_dir = ROOT / "data" / "array_raw"
array_geometry_dir = ROOT / "data" / "array_geometry"

phase3_src_dir = ROOT / "src" / "phase3_array"
hybrid_src_dir = ROOT / "src" / "hybrid_option2_option3"
phase2_harmonic_src_dir = ROOT / "src" / "phase2_harmonic_fusion"

option2_model_path = ROOT / "models" / "drone_cnn_phase2_v3_multiview_hardnegatives.pth"
option3_model_path = ROOT / "models" / "drone_cnn_phase2_v4_specialist_ensemble.pth"
hybrid_config_path = hybrid_src_dir / "config_hybrid.py"

phase2_harmonic_model_path = ROOT / "models" / "phase2_harmonic_fusion" / "drone_cnn_phase2_harmonic_fusion_v1.pth"
phase2_harmonic_backbone_path = ROOT / "models" / "phase2v5_real_noise" / "drone_cnn_phase2v5c_real_noise_balanced.pth"
phase2_harmonic_threshold = 0.85

results_dir = ROOT / "results" / "phase3_array"
models_dir = ROOT / "models" / "phase3_array"
beam_scan_results_dir = results_dir / "beam_scan_results"
plots_dir = results_dir / "plots"
logs_dir = results_dir / "logs"
comparisons_dir = results_dir / "comparisons"

single_channel_index = 0
warn_if_file_longer_sec = 30 * 60

hybrid_rule = "B"
hybrid_option3_score_method = "weighted_average"
hybrid_enable_veto = True


def window_samples():
    return int(round(window_sec * sample_rate_target))


def hop_samples():
    return int(round(hop_sec * sample_rate_target))


def ensure_output_dirs():
    for d in (
        results_dir,
        models_dir,
        beam_scan_results_dir,
        plots_dir,
        logs_dir,
        comparisons_dir,
        array_raw_dir,
        array_geometry_dir,
    ):
        d.mkdir(parents=True, exist_ok=True)
