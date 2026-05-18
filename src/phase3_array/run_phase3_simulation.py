"""Generate simulated array WAVs for Phase 3 testing.

Examples:
  python -m src.phase3_array.run_phase3_simulation
  python -m src.phase3_array.run_phase3_simulation --scenario drone_tank --az 90 --el 40 --duration 12
  python -m src.phase3_array.run_phase3_simulation --preset-suite
"""

import argparse
from datetime import datetime
from pathlib import Path

from . import config_phase3 as config
from .simulate_array_wav import make_controlled_source_bank, simulate_array_wav


def _safe_name(text: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in text)


def _make_one(args, scenario, az, el, suffix="", source_bank=None):
    name = args.out_name
    if name is None:
        tag = args.tag_prefix or f"{scenario}_az{int(az)}_el{int(el)}{suffix}"
        if args.tag_prefix:
            tag = f"{tag}_{scenario}{suffix}"
        safe_tag = _safe_name(tag)
        prefix = "" if safe_tag.startswith("array_sim_") else "array_sim_"
        name = f"{prefix}{safe_tag}.wav"
    out_wav = Path(config.array_raw_dir) / name
    wav, truth = simulate_array_wav(
        out_wav,
        scenario=scenario,
        az_deg=az,
        el_deg=el,
        duration_sec=args.duration,
        interferer_az_deg=args.interferer_az,
        interferer_el_deg=args.interferer_el,
        snr_db=args.snr_db,
        source_bank=source_bank,
    )
    print(f"Created: {wav}")
    print(f"Truth  : {truth}")


def main():
    parser = argparse.ArgumentParser(description="Generate simulated Phase 3 array WAVs")
    parser.add_argument(
        "--scenario",
        choices=["drone", "drone_tank", "drone_engine", "drone_crowd", "tank", "engine", "crowd", "noise", "pure_noise"],
        default="drone",
    )
    parser.add_argument("--az", type=float, default=90.0)
    parser.add_argument("--el", type=float, default=40.0)
    parser.add_argument("--duration", type=float, default=12.0)
    parser.add_argument("--interferer-az", type=float, default=225.0)
    parser.add_argument("--interferer-el", type=float, default=20.0)
    parser.add_argument("--snr-db", type=float, default=0.0)
    parser.add_argument("--out-name", type=str, default=None)
    parser.add_argument("--preset-suite", action="store_true")
    parser.add_argument("--controlled-suite", action="store_true", help="Generate paired scenarios with shared source waveforms.")
    parser.add_argument("--tag-prefix", type=str, default=None, help="Optional shared filename tag, e.g. array_sim_test_YYYYMMDD_HHMMSS.")
    args = parser.parse_args()

    config.ensure_output_dirs()
    Path(config.array_raw_dir).mkdir(parents=True, exist_ok=True)

    if args.controlled_suite:
        if args.tag_prefix is None:
            args.tag_prefix = "array_sim_test_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        n = int(round(args.duration * config.sample_rate_target))
        source_bank = make_controlled_source_bank(n, config.sample_rate_target, include=("drone",))
        presets = [
            ("drone", 90, 40, ""),
            ("drone_tank", 90, 40, "_tank0dB"),
            ("tank", 225, 20, ""),
        ]
        old_name = args.out_name
        for scenario, az, el, suffix in presets:
            args.out_name = None
            _make_one(args, scenario, az, el, suffix, source_bank=source_bank)
        args.out_name = old_name
    elif args.preset_suite:
        presets = [
            ("drone", 90, 40, ""),
            ("drone_tank", 90, 40, "_tank0dB"),
            ("drone_engine", 45, 40, ""),
            ("drone_crowd", 135, 40, ""),
            ("tank", 225, 20, ""),
            ("engine", 270, 20, ""),
            ("crowd", 180, 20, ""),
            ("pure_noise", 0, 20, ""),
        ]
        old_name = args.out_name
        for scenario, az, el, suffix in presets:
            args.out_name = None
            _make_one(args, scenario, az, el, suffix)
        args.out_name = old_name
    else:
        _make_one(args, args.scenario, args.az, args.el)

    print()
    print("Next:")
    print("  python -m src.phase3_array.run_phase3_array")


if __name__ == "__main__":
    main()
