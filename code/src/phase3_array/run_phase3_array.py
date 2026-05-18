"""Main Phase 3 passive array runner.

Run:
  python -m src.phase3_array.run_phase3_array
"""

from pathlib import Path

import torch

from . import config_phase3 as config
from .compare_array_vs_single_channel import compare_array_vs_single_channel
from .evaluate_array_file import evaluate_array_wav
from .hybrid_detector_wrapper import load_hybrid_detector


def _find_array_wavs():
    return sorted(Path(config.array_raw_dir).glob("*.wav"))


def _print_header():
    print("=" * 76)
    print("Phase 3: Passive microphone-array beamforming + hybrid detector")
    print("=" * 76)
    print("Safety:")
    print("  - No CNN training will be performed.")
    print("  - No existing Option 2 / Option 3 / hybrid models will be overwritten.")
    print("  - All Phase 3 outputs are saved under results/phase3_array/.")
    print()
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"GPU   : {props.name} ({props.total_memory/1e9:.1f} GB)")
    print(f"Array WAV folder : {config.array_raw_dir}")
    print(f"Geometry mode    : {config.geometry_mode}")
    print(f"Azimuth grid     : {config.azimuth_grid_deg}")
    print(f"Elevation grid   : {config.elevation_grid_deg}")
    print(f"Energy prefilter : {config.use_energy_prefilter} top_k={config.top_k_beams_for_hybrid}")
    print(f"Smoothing        : {config.smoothing_mode}")
    print(f"Detector mode    : {getattr(config, 'detector_mode', 'hybrid_option2_option3')}")
    print()


def _file_verdict(summary, comparison):
    beam_rate = comparison.get("beamformed_smoothed_detection_rate_percent", 0.0)
    single_rate = comparison.get("single_channel_detection_rate_percent", 0.0)
    delta = comparison.get("mean_score_delta_beam_minus_single", 0.0)
    if delta > 0.02 or beam_rate > single_rate:
        helped = "beamforming helped"
    elif delta < -0.02 or beam_rate < single_rate:
        helped = "beamforming did not help"
    else:
        helped = "beamforming maintained performance"
    direction = summary.get("most_common_direction")
    intervals = summary.get("detection_intervals", [])
    return helped, direction, intervals


def main():
    config.ensure_output_dirs()
    _print_header()

    wavs = _find_array_wavs()
    if not wavs:
        print("No multichannel array WAV files found. Place files in data/array_raw/.")
        return

    hybrid_detector = load_hybrid_detector(config)
    print("Loaded detector:")
    if hybrid_detector.get("detector_mode") == "phase2_harmonic":
        print(f"  Phase 2 harmonic: {config.phase2_harmonic_model_path.name}")
        print(f"  Backbone        : {config.phase2_harmonic_backbone_path.name}")
        print(f"  Threshold       : {config.phase2_harmonic_threshold}")
    else:
        print(f"  Option 2: {config.option2_model_path.name}")
        print(f"  Option 3: {config.option3_model_path.name}")
    print()

    final_rows = []
    for idx, wav_path in enumerate(wavs, 1):
        print(f"[{idx}/{len(wavs)}] Evaluating {wav_path.name}")
        try:
            summary = evaluate_array_wav(wav_path, config, hybrid_detector=hybrid_detector)
            comparison = compare_array_vs_single_channel(
                wav_path,
                summary["per_window_csv"],
                config,
                hybrid_detector=hybrid_detector,
            )
            helped, direction, intervals = _file_verdict(summary, comparison)
            final_rows.append((wav_path.name, helped, direction, intervals, summary, comparison))
            print(f"  Verdict   : {helped}")
            if direction:
                print(f"  Direction : az={direction['az']} el={direction['el']} count={direction['count']}")
            print(f"  Intervals : {intervals if intervals else 'none'}")
            print(f"  Scores    : single_mean={comparison['mean_single_channel_score']:.3f} "
                  f"beam_mean={comparison['mean_beamformed_score']:.3f}")
        except Exception as e:
            print(f"  ERROR: {e}")
        print()

    print("=" * 76)
    print("Phase 3 final summary")
    print("=" * 76)
    for name, helped, direction, intervals, summary, comparison in final_rows:
        dtext = "unknown"
        if direction:
            dtext = f"az={direction['az']} el={direction['el']}"
        print(f"{name}: {helped}; best direction {dtext}; intervals={intervals if intervals else 'none'}")
    print()
    print("Recommended next action:")
    print("  Add/verify real multichannel WAVs and custom mic geometry, then compare direction stability.")


if __name__ == "__main__":
    main()
