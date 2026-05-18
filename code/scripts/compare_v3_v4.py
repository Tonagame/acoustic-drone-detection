"""
compare_v3_v4.py
----------------
Side-by-side comparison of:
  v3: one multiview CNN with filter augmentation
  v4: five fixed-view specialist CNNs in one ensemble bundle

Usage:
  python compare_v3_v4.py
  python compare_v3_v4.py --chunks 200
  python compare_v3_v4.py --no-gpu
"""

import argparse
import random
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from train_phase2_v4_specialist import (
    DRONE_DIR,
    FMAX_THR,
    FS,
    MODELS_DIR,
    SCORE_THR,
    VIEW_NAMES,
    VIEW_WEIGHTS,
    VOTE_THR,
    VOTES_NEED,
    WIN_SAMPLES,
    DroneCNN,
    audio_to_logmel,
    create_audio_views,
    load_wav,
    mix_at_snr,
    synth_crowd,
    synth_engine,
    synth_pure_noise,
    synth_tank,
    window_audio,
)

V3_PATH = MODELS_DIR / "drone_cnn_phase2_v3_multiview_hardnegatives.pth"
V4_PATH = MODELS_DIR / "drone_cnn_phase2_v4_specialist_ensemble.pth"


def load_v3(path: Path, device):
    ckpt = torch.load(str(path), map_location=device, weights_only=False)
    classes = ckpt.get("classes", ["drone", "no_drone"])
    model = DroneCNN(len(classes)).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    drone_idx = classes.index("drone") if "drone" in classes else 0
    return model, drone_idx


def load_v4(path: Path, device):
    bundle = torch.load(str(path), map_location=device, weights_only=False)
    drone_idx = int(bundle.get("drone_idx", 0))
    models = []
    for vi, vname in enumerate(bundle.get("view_names", VIEW_NAMES)):
        key = f'model_{vi}_{vname.replace("-","_").replace("+","_")}'
        if key not in bundle:
            matches = [k for k in bundle if k.startswith(f"model_{vi}_")]
            if not matches:
                raise KeyError(f"Missing v4 state dict for view {vi} ({vname})")
            key = matches[0]
        model = DroneCNN(2).to(device)
        model.load_state_dict(bundle[key])
        model.eval()
        models.append(model)
    return models, drone_idx


@torch.no_grad()
def score_v3(model, drone_idx, audio, device):
    views = create_audio_views(audio)
    probs = np.zeros(5, dtype=np.float32)
    for vi, view in enumerate(views):
        lm = audio_to_logmel(view)
        X = lm.unsqueeze(0).unsqueeze(0).to(device)
        sc = torch.softmax(model(X), dim=1)
        probs[vi] = sc[0, drone_idx].item()
    return score_from_probs(probs)


@torch.no_grad()
def score_v4(models, drone_idx, audio, device):
    views = create_audio_views(audio)
    probs = np.zeros(5, dtype=np.float32)
    for vi, (model, view) in enumerate(zip(models, views)):
        lm = audio_to_logmel(view)
        X = lm.unsqueeze(0).unsqueeze(0).to(device)
        sc = torch.softmax(model(X), dim=1)
        probs[vi] = sc[0, drone_idx].item()
    return score_from_probs(probs)


def score_from_probs(probs):
    ws = float(VIEW_WEIGHTS @ probs)
    fm = float(probs[1:].max())
    vc = int((probs > VOTE_THR).sum())
    det = (fm > FMAX_THR) or (ws > SCORE_THR) or (vc >= VOTES_NEED)
    return ws, fm, vc, det, probs


def build_drone_window_pool(max_files=1500):
    files = [f for f in sorted(DRONE_DIR.glob("*.wav"))
             if sf.info(str(f)).frames >= WIN_SAMPLES]
    if max_files and len(files) > max_files:
        files = random.sample(files, max_files)
    wins = []
    for f in files:
        try:
            wins.extend(window_audio(load_wav(f)))
        except Exception:
            pass
    random.shuffle(wins)
    return wins


def make_next_drone(wins):
    pos = [0]
    def next_drone():
        if pos[0] >= len(wins):
            random.shuffle(wins)
            pos[0] = 0
        w = wins[pos[0]]
        pos[0] += 1
        return w.copy()
    return next_drone


def evaluate_scenario(name, expect_detect, gen_fn, v3, v4, device, chunks):
    v3_model, v3_idx = v3
    v4_models, v4_idx = v4
    v3_dets, v4_dets = [], []
    v3_ws, v4_ws = [], []

    for i in range(chunks):
        audio = gen_fn(i)
        ws3, _, _, det3, _ = score_v3(v3_model, v3_idx, audio, device)
        ws4, _, _, det4, _ = score_v4(v4_models, v4_idx, audio, device)
        v3_dets.append(det3)
        v4_dets.append(det4)
        v3_ws.append(ws3)
        v4_ws.append(ws4)

    v3_rate = float(np.mean(v3_dets) * 100.0)
    v4_rate = float(np.mean(v4_dets) * 100.0)
    v3_mean = float(np.mean(v3_ws))
    v4_mean = float(np.mean(v4_ws))
    winner = pick_winner(expect_detect, v3_rate, v3_mean, v4_rate, v4_mean)
    return name, v3_rate, v3_mean, v4_rate, v4_mean, winner


def pick_winner(expect_detect, v3_rate, v3_ws, v4_rate, v4_ws):
    if expect_detect:
        if abs(v4_rate - v3_rate) > 1.0:
            return "v4" if v4_rate > v3_rate else "v3"
        if abs(v4_ws - v3_ws) > 0.02:
            return "v4" if v4_ws > v3_ws else "v3"
        return "tie"
    if abs(v4_rate - v3_rate) > 1.0:
        return "v4" if v4_rate < v3_rate else "v3"
    if abs(v4_ws - v3_ws) > 0.02:
        return "v4" if v4_ws < v3_ws else "v3"
    return "tie"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", type=int, default=600,
                        help="One-second windows per scenario")
    parser.add_argument("--no-gpu", action="store_true")
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_gpu else "cpu")

    if not V3_PATH.exists():
        raise FileNotFoundError(f"Missing v3 model: {V3_PATH}")
    if not V4_PATH.exists():
        raise FileNotFoundError(
            f"Missing v4 bundle: {V4_PATH}\nRun: python train_phase2_v4_specialist.py --quick")

    print("=" * 78)
    print("  Phase 2v3 vs Phase 2v4 specialist ensemble")
    print("=" * 78)
    print(f"  Device  : {device}")
    print(f"  v3      : {V3_PATH.name}")
    print(f"  v4      : {V4_PATH.name}")
    print(f"  Views   : {VIEW_NAMES}")
    print(f"  Weights : {VIEW_WEIGHTS.tolist()}")
    print(f"  Windows : {args.chunks} per scenario\n")

    v3 = load_v3(V3_PATH, device)
    v4 = load_v4(V4_PATH, device)

    drone_wins = build_drone_window_pool()
    if not drone_wins:
        raise RuntimeError(f"No >=1s drone WAV windows found under {DRONE_DIR}")

    next_drone = make_next_drone(drone_wins)
    scenarios = [
        ("drone alone",  True,  lambda i: next_drone()),
        ("drone+tank",   True,  lambda i: mix_at_snr(next_drone(), synth_tank(WIN_SAMPLES, i), 0)),
        ("drone+engine", True,  lambda i: mix_at_snr(next_drone(), synth_engine(WIN_SAMPLES, i), 0)),
        ("drone+crowd",  True,  lambda i: mix_at_snr(next_drone(), synth_crowd(WIN_SAMPLES, i), 0)),
        ("tank alone",   False, lambda i: synth_tank(WIN_SAMPLES, i)),
        ("engine alone", False, lambda i: synth_engine(WIN_SAMPLES, i)),
        ("crowd alone",  False, lambda i: synth_crowd(WIN_SAMPLES, i)),
        ("pure noise",   False, lambda i: synth_pure_noise(WIN_SAMPLES, i)),
    ]

    print("Scenario               | v3 Det% | v3 ws  | v4 Det% | v4 ws  | Winner")
    print("-----------------------+---------+--------+---------+--------+-------")
    rows = []
    for name, expect, gen_fn in scenarios:
        row = evaluate_scenario(name, expect, gen_fn, v3, v4, device, args.chunks)
        rows.append(row)
        _, v3_rate, v3_mean, v4_rate, v4_mean, winner = row
        print(f"{name:<22s} | {v3_rate:6.1f}% | {v3_mean:6.3f} | "
              f"{v4_rate:6.1f}% | {v4_mean:6.3f} | {winner}")

    print()


if __name__ == "__main__":
    main()
