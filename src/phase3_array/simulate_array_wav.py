"""Generate simulated multichannel microphone-array WAV files.

The simulator creates far-field plane-wave sources by applying per-microphone
TDOA delays from the configured array geometry. It is intentionally simple:
no MVDR, no room model, no per-channel filtering beyond gain/noise jitter.
"""

import json
import random
import uuid
from pathlib import Path

import numpy as np
import soundfile as sf

from . import config_phase3 as config
from .array_geometry import load_array_geometry
from .delay_and_sum import compute_delays
from .direction_grid import unit_vector_from_az_el
from .fractional_delay import apply_fractional_delay

_STRONG_DRONE_CACHE = {}


def _norm(x: np.ndarray, level=0.85) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    x = x - float(np.mean(x))
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    return (x / peak * level).astype(np.float32) if peak > 1e-7 else x


def _lp(s, taps):
    return np.convolve(s, np.ones(taps) / taps, mode="same")


def _concat_with_crossfade(chunks: list[np.ndarray], n: int, fade_len: int) -> np.ndarray:
    if not chunks:
        raise ValueError("No chunks to concatenate.")
    out = chunks[0].astype(np.float32).copy()
    for chunk in chunks[1:]:
        chunk = chunk.astype(np.float32)
        if fade_len > 0 and len(out) >= fade_len and len(chunk) >= fade_len:
            fade = np.linspace(0.0, 1.0, fade_len, endpoint=False, dtype=np.float32)
            out[-fade_len:] = out[-fade_len:] * (1.0 - fade) + chunk[:fade_len] * fade
            out = np.concatenate([out, chunk[fade_len:]])
        else:
            out = np.concatenate([out, chunk])
        if len(out) >= n:
            break
    if len(out) < n:
        reps = int(np.ceil(n / max(len(out), 1)))
        out = np.tile(out, reps)
    return out[:n].astype(np.float32)


def synth_drone_like(n, fs, t0=0.0):
    """Fallback synthetic drone-like harmonic buzz."""
    rng = np.random.default_rng(int(t0 * 1000 + 11) % 99991)
    f0 = rng.uniform(170.0, 260.0)
    t = np.linspace(t0, t0 + n / fs, n, endpoint=False)
    wobble = 1.0 + 0.025 * np.sin(2 * np.pi * 3.5 * t)
    ph = np.cumsum(wobble) * (f0 / fs) * 2 * np.pi
    sig = (
        0.50 * np.sin(ph)
        + 0.25 * np.sin(2 * ph)
        + 0.12 * np.sin(3 * ph)
        + 0.06 * np.sin(4 * ph)
    )
    sig += 0.04 * rng.standard_normal(n)
    return _norm(sig)


def synth_tank(n, fs, t0=0.0):
    t = np.linspace(t0, t0 + n / fs, n, endpoint=False)
    rpm = 1.0 + 0.04 * np.sin(2 * np.pi * 0.3 * t)
    f0 = 45.0
    eng = (
        0.55 * np.sin(2 * np.pi * f0 * rpm * t)
        + 0.25 * np.sin(2 * np.pi * f0 * 2 * rpm * t)
        + 0.12 * np.sin(2 * np.pi * f0 * 3 * rpm * t)
        + 0.08 * np.sin(2 * np.pi * f0 * 4 * rpm * t)
    )
    rng = np.random.default_rng(int(t0 * 100) % 9999)
    clank = np.zeros(n)
    for pos in range(0, n, int(fs * 0.15)):
        b = min(int(fs * 0.01), n - pos)
        clank[pos:pos + b] = rng.standard_normal(b) * 0.4
    return _norm(eng + clank + _lp(rng.standard_normal(n), 64) * 0.3)


def synth_engine(n, fs, t0=0.0):
    rng = np.random.default_rng(int(t0 * 1000 + 17) % 99991)
    f0 = rng.uniform(60.0, 120.0)
    t = np.linspace(t0, t0 + n / fs, n, endpoint=False)
    rpm = 1.0 + 0.05 * np.sin(2 * np.pi * 1.2 * t)
    ph = np.cumsum(rpm) * (f0 / fs) * 2 * np.pi
    harm = (
        0.55 * np.sin(ph)
        + 0.25 * np.sin(2 * ph)
        + 0.12 * np.sin(3 * ph)
        + 0.06 * np.sin(4 * ph)
        + 0.03 * np.sin(5 * ph)
    )
    exhaust = _lp(rng.standard_normal(n), max(1, int(fs / 2000))) * 0.7
    return _norm(harm + exhaust)


def synth_crowd(n, fs, t0=0.0):
    rng = np.random.default_rng(int(t0 * 1000 + 23) % 99991)
    white = rng.standard_normal(n)
    bp = _lp(white - _lp(white, 80), 5)
    t = np.linspace(t0, t0 + n / fs, n, endpoint=False)
    am = 0.4 + 0.6 * np.abs(np.sin(2 * np.pi * 3.0 * t))
    return _norm(bp * am, 0.6)


def synth_noise(n, fs, t0=0.0):
    rng = np.random.default_rng(int(t0 * 1000 + 71) % 99991)
    return _norm(rng.standard_normal(n), 0.45)


def _load_real_drone_or_fallback(n, fs):
    drone_dir = config.ROOT / "data" / "raw" / "drone"
    files = []
    for p in sorted(drone_dir.glob("*.wav")):
        try:
            if sf.info(str(p)).frames >= fs:
                files.append(p)
        except Exception:
            pass
    random.shuffle(files)
    chunks = []
    total = 0
    for path in files[:2000]:
        try:
            audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if sr != fs:
                continue
            if len(audio) < fs:
                continue
            if len(audio) > fs:
                start = random.randint(0, len(audio) - fs)
                audio = audio[start:start + fs]
            audio = _norm(audio, 0.85)
            chunks.append(audio)
            total += len(audio)
            if total >= n:
                break
        except Exception:
            pass
    if chunks and total >= n:
        return _concat_with_crossfade(chunks, n, max(1, int(0.02 * fs)))
    return synth_drone_like(n, fs)


def _load_strong_real_drone_or_fallback(n, fs, min_score=0.70, max_candidates=500):
    """
    Build simulation drone audio from real clips that the current hybrid
    detector recognizes. This keeps the array simulator focused on testing
    beamforming/directionality rather than random weak dataset snippets.
    """
    cache_key = (fs, round(min_score, 2), max_candidates)
    if cache_key in _STRONG_DRONE_CACHE:
        chunks = _STRONG_DRONE_CACHE[cache_key]
        if chunks:
            return _concat_with_crossfade(chunks, n, max(1, int(0.02 * fs)))

    try:
        from .hybrid_detector_wrapper import load_hybrid_detector, predict_hybrid_on_mono_window

        hybrid = load_hybrid_detector(config)
        drone_dir = config.ROOT / "data" / "raw" / "drone"
        files = []
        for p in sorted(drone_dir.glob("*.wav")):
            try:
                if sf.info(str(p)).frames >= fs:
                    files.append(p)
            except Exception:
                pass
        random.shuffle(files)

        scored = []
        for path in files[:max_candidates]:
            try:
                audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)
                if sr != fs or len(audio) < fs:
                    continue
                if len(audio) > fs:
                    start = random.randint(0, len(audio) - fs)
                    audio = audio[start:start + fs]
                audio = _norm(audio, 0.85)
                score, detected, _ = predict_hybrid_on_mono_window(audio, fs, hybrid)
                if detected or score >= min_score:
                    scored.append((float(score), audio.astype(np.float32)))
                if len(scored) >= max(12, int(np.ceil(n / fs)) * 2):
                    break
            except Exception:
                pass
        scored.sort(key=lambda item: item[0], reverse=True)
        chunks = [audio for _, audio in scored[:max(12, int(np.ceil(n / fs)) * 2)]]
        _STRONG_DRONE_CACHE[cache_key] = chunks
        if chunks:
            return _concat_with_crossfade(chunks, n, max(1, int(0.02 * fs)))
    except Exception:
        pass

    return _load_real_drone_or_fallback(n, fs)


def make_mono_source(kind: str, n: int, fs: int, t0=0.0) -> np.ndarray:
    kind = kind.lower()
    if kind == "drone":
        return _load_strong_real_drone_or_fallback(n, fs)
    if kind == "tank":
        return synth_tank(n, fs, t0)
    if kind == "engine":
        return synth_engine(n, fs, t0)
    if kind == "crowd":
        return synth_crowd(n, fs, t0)
    if kind in ("noise", "pure_noise"):
        return synth_noise(n, fs, t0)
    raise ValueError(f"Unknown source kind: {kind}")


def make_controlled_source_bank(
    n: int,
    fs: int,
    include: tuple[str, ...] = ("drone", "tank", "engine", "crowd"),
) -> dict:
    """
    Build reusable mono sources for controlled A/B simulations.

    Scenarios that share this bank use the exact same drone waveform and level,
    so "drone" vs "drone+tank" differs only by the added interferer.
    """
    bank = {}
    source_id = uuid.uuid4().hex[:12]
    for kind in include:
        bank[kind] = {
            "audio": make_mono_source(kind, n, fs, t0=0.0),
            "source_id": f"{source_id}_{kind}",
        }
    return bank


def _bank_audio(source_bank: dict | None, kind: str, n: int, fs: int, t0=0.0) -> tuple[np.ndarray, str | None]:
    if source_bank and kind in source_bank:
        item = source_bank[kind]
        if isinstance(item, dict):
            audio = np.asarray(item["audio"], dtype=np.float32)
            source_id = item.get("source_id")
        else:
            audio = np.asarray(item, dtype=np.float32)
            source_id = None
        if len(audio) < n:
            reps = int(np.ceil(n / max(len(audio), 1)))
            audio = np.tile(audio, reps)
        return audio[:n].astype(np.float32), source_id
    return make_mono_source(kind, n, fs, t0=t0), None


def plane_wave_to_array(
    mono: np.ndarray,
    fs: int,
    mic_positions: np.ndarray,
    az_deg: float,
    el_deg: float,
    speed_of_sound: float = 343.0,
    source_level: float = 0.75,
    sensor_noise_level: float = 0.0005,
    gain_jitter: float = 0.02,
) -> tuple[np.ndarray, dict]:
    unit = unit_vector_from_az_el(az_deg, el_deg)
    delays_sec = compute_delays(mic_positions, unit, speed_of_sound)
    delays_samples = delays_sec * fs
    rng = np.random.default_rng(2026)
    chans = []
    for delay in delays_samples:
        ch = apply_fractional_delay(mono, float(delay))
        ch *= source_level * rng.uniform(1.0 - gain_jitter, 1.0 + gain_jitter)
        if sensor_noise_level > 0:
            ch += rng.standard_normal(len(ch)).astype(np.float32) * sensor_noise_level
        chans.append(ch.astype(np.float32))
    x = np.stack(chans, axis=1)
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak > 0.98:
        x = x / peak * 0.98
    return x.astype(np.float32), {
        "az_deg": float(az_deg),
        "el_deg": float(el_deg),
        "delays_sec": delays_sec.tolist(),
        "delays_samples": delays_samples.tolist(),
    }


def diffuse_source_to_array(
    kind: str,
    n: int,
    fs: int,
    mic_positions: np.ndarray,
    center_az_deg: float,
    center_el_deg: float,
    speed_of_sound: float = 343.0,
    source_level: float = 0.55,
    n_subsources: int = 5,
    spread_az_deg: float = 55.0,
    spread_el_deg: float = 20.0,
    sensor_noise_level: float = 0.001,
    base_audio: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    """
    Simulate a less coherent negative source.

    Heavy vehicles and crowds are not ideal plane waves in this toy sim: they
    contain several vibrating/noisy parts plus ground reflections. We approximate
    that with multiple nearby sources and extra per-channel noise.
    """
    stable_kind_id = sum((i + 1) * ord(c) for i, c in enumerate(kind))
    rng = np.random.default_rng(7701 + stable_kind_id)
    parts = []
    sub_truth = []
    for si in range(n_subsources):
        az = (center_az_deg + rng.uniform(-spread_az_deg, spread_az_deg)) % 360.0
        el = float(np.clip(center_el_deg + rng.uniform(-spread_el_deg, spread_el_deg), 0.0, 85.0))
        if base_audio is None:
            mono = make_mono_source(kind, n, fs, t0=si * 0.37)
        else:
            shift = int(round(si * 0.037 * fs)) % max(1, len(base_audio))
            mono = np.roll(np.asarray(base_audio, dtype=np.float32), shift)[:n]
        level = source_level / np.sqrt(n_subsources) * rng.uniform(0.75, 1.15)
        x_part, dbg = plane_wave_to_array(
            mono,
            fs,
            mic_positions,
            az,
            el,
            speed_of_sound=speed_of_sound,
            source_level=level,
            sensor_noise_level=0.0,
            gain_jitter=0.08,
        )
        parts.append(x_part)
        sub_truth.append({"kind": kind, "subsource": si, **dbg})
    x = np.sum(np.stack(parts, axis=0), axis=0).astype(np.float32)
    if sensor_noise_level > 0:
        x += rng.standard_normal(x.shape).astype(np.float32) * sensor_noise_level
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak > 0.98:
        x = x / peak * 0.98
    return x.astype(np.float32), {
        "az_deg": float(center_az_deg),
        "el_deg": float(center_el_deg),
        "diffuse": True,
        "n_subsources": int(n_subsources),
        "subsources": sub_truth,
    }


def mix_sources(sources: list[np.ndarray]) -> np.ndarray:
    if not sources:
        raise ValueError("At least one source is required.")
    x = np.sum(np.stack(sources, axis=0), axis=0)
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak > 0.98:
        x = x / peak * 0.98
    return x.astype(np.float32)


def simulate_array_wav(
    out_wav: Path,
    scenario: str = "drone",
    az_deg: float = 90.0,
    el_deg: float = 40.0,
    duration_sec: float = 12.0,
    interferer_az_deg: float = 225.0,
    interferer_el_deg: float = 20.0,
    snr_db: float = 0.0,
    source_bank: dict | None = None,
):
    config.ensure_output_dirs()
    out_wav = Path(out_wav)
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    fs = config.sample_rate_target
    n = int(round(duration_sec * fs))
    mic_positions = load_array_geometry(config)

    scenario = scenario.lower()
    sources = []
    truth_sources = []

    if scenario in ("drone", "drone_tank", "drone_engine", "drone_crowd"):
        drone, source_id = _bank_audio(source_bank, "drone", n, fs)
        x_drone, dbg = plane_wave_to_array(
            drone, fs, mic_positions, az_deg, el_deg,
            speed_of_sound=config.speed_of_sound,
            source_level=0.75,
        )
        sources.append(x_drone)
        truth_sources.append({"kind": "drone", "source_id": source_id, "source_level": 0.75, **dbg})

    if scenario == "drone_tank":
        level = 0.75 * (10 ** (-snr_db / 20.0))
        x_noise, dbg = diffuse_source_to_array(
            "tank", n, fs, mic_positions, interferer_az_deg, interferer_el_deg,
            speed_of_sound=config.speed_of_sound,
            source_level=level,
            n_subsources=7,
            spread_az_deg=70.0,
            spread_el_deg=25.0,
        )
        sources.append(x_noise)
        truth_sources.append({"kind": "tank", "source_id": None, "source_level": level, **dbg, "snr_db": float(snr_db)})
    elif scenario == "drone_engine":
        x_noise, dbg = diffuse_source_to_array(
            "engine", n, fs, mic_positions, interferer_az_deg, interferer_el_deg,
            speed_of_sound=config.speed_of_sound,
            source_level=0.75,
            n_subsources=5,
            spread_az_deg=45.0,
            spread_el_deg=18.0,
        )
        sources.append(x_noise)
        truth_sources.append({"kind": "engine", "source_id": None, "source_level": 0.75, **dbg})
    elif scenario == "drone_crowd":
        noise, crowd_source_id = _bank_audio(source_bank, "crowd", n, fs)
        x_noise, dbg = plane_wave_to_array(
            noise, fs, mic_positions, interferer_az_deg, interferer_el_deg,
            speed_of_sound=config.speed_of_sound,
            source_level=0.75,
        )
        sources.append(x_noise)
        truth_sources.append({"kind": "crowd", "source_id": crowd_source_id, "source_level": 0.75, **dbg})
    elif scenario in ("tank", "engine", "crowd", "noise", "pure_noise"):
        kind = "noise" if scenario == "pure_noise" else scenario
        if kind in ("tank", "engine", "crowd"):
            x_src, dbg = diffuse_source_to_array(
                kind, n, fs, mic_positions, az_deg, el_deg,
                speed_of_sound=config.speed_of_sound,
                source_level=0.65,
                n_subsources=7 if kind == "tank" else 5,
                spread_az_deg=70.0 if kind == "tank" else 45.0,
                spread_el_deg=25.0 if kind == "tank" else 18.0,
            )
            source_id = None
        else:
            mono, source_id = _bank_audio(source_bank, kind, n, fs)
            x_src, dbg = plane_wave_to_array(
                mono, fs, mic_positions, az_deg, el_deg,
                speed_of_sound=config.speed_of_sound,
                source_level=0.45,
                sensor_noise_level=0.02,
            )
        sources.append(x_src)
        truth_sources.append({"kind": kind, "source_id": source_id, **dbg})

    if not sources:
        raise ValueError(f"Unsupported scenario: {scenario}")

    x = mix_sources(sources)
    sf.write(str(out_wav), x, fs)

    truth = {
        "wav_path": str(out_wav),
        "scenario": scenario,
        "sample_rate": fs,
        "duration_sec": float(duration_sec),
        "num_channels": int(x.shape[1]),
        "geometry_mode": config.geometry_mode,
        "mic_spacing_m": config.mic_spacing_m,
        "mic_positions": mic_positions.tolist(),
        "sources": truth_sources,
        "primary_az_deg": float(az_deg),
        "primary_el_deg": float(el_deg),
    }
    truth_path = out_wav.with_suffix(".json")
    truth_path.write_text(json.dumps(truth, indent=2), encoding="utf-8")
    return out_wav, truth_path
