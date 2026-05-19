from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.phase2v5_real_noise.audio_phase2v5 import AudioPreprocessor
from src.phase2v5_real_noise.config_phase2v5 import DRONE_DIR, FSD50K_CANDIDATES_CSV, FSD_LABELS, NODRONE_DIR
from src.phase2v5_real_noise.data_phase2v5 import AudioFileWindowPool, FSD50KWindowPool
from src.phase2v5_real_noise.model_phase2v5 import FocalLoss
from src.phase3_real_noise_specialists.predict_phase3_hybrid import load_phase2_guard, load_specialist_bundle, predict_phase2_guard

from src.phase3_mid_fusion.config_mid_fusion import (
    FEATURES_DIR,
    GUARD_FEATURE_NAMES,
    GUARD_NECK_INPUT_DIM,
    GUARD_NECK_SAVE_PATH,
    PHASE2_BACKBONE_PATH,
    PHASE2_GUARD_PATH,
    RESULTS_DIR,
    SNR_LEVELS,
    SPECIALIST_BUNDLE_PATH,
    VIEW_NAMES,
    ensure_dirs,
)
from src.phase3_mid_fusion.model_mid_fusion import GuardNeckFusionHead


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_pools(args, preproc):
    drone_pool = AudioFileWindowPool(DRONE_DIR, preproc, args.max_drone_files, args.seed)
    nodrone_pool = AudioFileWindowPool(NODRONE_DIR, preproc, args.max_nodrone_files, args.seed + 1) if NODRONE_DIR.exists() else None
    fsd_pool = FSD50KWindowPool(FSD50K_CANDIDATES_CSV, preproc, FSD_LABELS, args.max_fsd_clips_per_label, args.seed + 2)
    return drone_pool, fsd_pool, nodrone_pool


def sample_train_audio(idx, examples_per_class, pools, preproc, rng, snr_levels, positive_mix_prob, negative_fsd_prob):
    drone_pool, fsd_pool, nodrone_pool = pools
    is_drone = idx < examples_per_class
    if is_drone:
        drone = drone_pool.sample_window(rng)
        if rng.random() < positive_mix_prob:
            return preproc.mix_at_snr(drone, fsd_pool.sample_window(rng), rng.choice(snr_levels)), 0, "drone_plus_fsd"
        return drone, 0, "drone_alone"
    use_fsd = nodrone_pool is None or rng.random() < negative_fsd_prob
    if use_fsd:
        return fsd_pool.sample_window(rng), 1, "fsd_alone"
    return nodrone_pool.sample_window(rng), 1, "dads_no_drone"


@torch.no_grad()
def specialist_latent(specialists_bundle, preproc: AudioPreprocessor, audio: np.ndarray):
    device = specialists_bundle["device"]
    views = preproc.create_audio_views(audio)
    latents = []
    for vi, (model, view) in enumerate(zip(specialists_bundle["models"], views)):
        lm = preproc.audio_to_logmel(view).unsqueeze(0).unsqueeze(0).to(device)
        latents.append(model.encode(lm).squeeze(0).detach().cpu().numpy())
    return np.concatenate(latents).astype(np.float32)


def guard_features(guard, preproc: AudioPreprocessor, audio: np.ndarray):
    gd = predict_phase2_guard(guard, preproc, audio)
    return np.asarray([gd.score, gd.vehicle_risk_score, gd.f0_norm, gd.harmonicity_score], dtype=np.float32)


def build_feature_cache(out_path: Path, args, device, n_per_class: int, seed: int):
    preproc = AudioPreprocessor(16000)
    pools = build_pools(args, preproc)
    specialists = load_specialist_bundle(args.specialists, device)
    guard = load_phase2_guard(args.phase2, args.backbone, device)
    rng = random.Random(seed)
    total = n_per_class * 2
    X = np.zeros((total, GUARD_NECK_INPUT_DIM), dtype=np.float32)
    y = np.zeros(total, dtype=np.int64)
    conditions = []
    t0 = time.perf_counter()
    print(f"Building guard-neck feature cache -> {out_path}")
    for idx in range(total):
        audio, label, condition = sample_train_audio(
            idx,
            n_per_class,
            pools,
            preproc,
            rng,
            SNR_LEVELS,
            args.positive_mix_prob,
            args.negative_fsd_prob,
        )
        X[idx, :320] = specialist_latent(specialists, preproc, audio)
        X[idx, 320:] = guard_features(guard, preproc, audio)
        y[idx] = label
        conditions.append(condition)
        if (idx + 1) % 500 == 0 or idx + 1 == total:
            print(f"  features {idx + 1:5d}/{total} elapsed={time.perf_counter() - t0:.1f}s")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        X=X,
        y=y,
        condition=np.asarray(conditions),
        feature_names=np.asarray([*(f"latent_{i}" for i in range(320)), *GUARD_FEATURE_NAMES]),
    )
    return out_path


def run_epoch(model, loader, criterion, device, optimizer=None):
    train = optimizer is not None
    model.train(train)
    loss_sum = 0.0
    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        if train:
            optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        if train:
            loss.backward()
            optimizer.step()
        loss_sum += float(loss.item()) * y.size(0)
        correct += int((logits.argmax(1) == y).sum().item())
        total += int(y.size(0))
    return loss_sum / max(total, 1), correct / max(total, 1)


def parse_args():
    ap = argparse.ArgumentParser(description="Train guard-neck mid fusion v2")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--examples-per-class", type=int, default=2500)
    ap.add_argument("--val-per-class", type=int, default=700)
    ap.add_argument("--epochs", type=int, default=45)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=7272)
    ap.add_argument("--torch-threads", type=int, default=3)
    ap.add_argument("--no-gpu", action="store_true")
    ap.add_argument("--max-drone-files", type=int, default=12000)
    ap.add_argument("--max-nodrone-files", type=int, default=5000)
    ap.add_argument("--max-fsd-clips-per-label", type=int, default=500)
    ap.add_argument("--positive-mix-prob", type=float, default=0.95)
    ap.add_argument("--negative-fsd-prob", type=float, default=0.95)
    ap.add_argument("--specialists", type=Path, default=SPECIALIST_BUNDLE_PATH)
    ap.add_argument("--phase2", type=Path, default=PHASE2_GUARD_PATH)
    ap.add_argument("--backbone", type=Path, default=PHASE2_BACKBONE_PATH)
    ap.add_argument("--train-cache", type=Path, default=None)
    ap.add_argument("--val-cache", type=Path, default=None)
    ap.add_argument("--rebuild-features", action="store_true")
    ap.add_argument("--save-path", type=Path, default=GUARD_NECK_SAVE_PATH)
    return ap.parse_args()


def main():
    args = parse_args()
    if args.quick:
        args.examples_per_class = min(args.examples_per_class, 300)
        args.val_per_class = min(args.val_per_class, 120)
        args.epochs = min(args.epochs, 16)
        args.max_drone_files = min(args.max_drone_files, 900)
        args.max_nodrone_files = min(args.max_nodrone_files, 600)
        args.max_fsd_clips_per_label = min(args.max_fsd_clips_per_label, 30)
        args.save_path = args.save_path.with_name(args.save_path.stem + "_quick" + args.save_path.suffix)
    ensure_dirs()
    torch.set_num_threads(max(1, args.torch_threads))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_gpu else "cpu")
    tag = "quick" if args.quick else "v2"
    train_cache = args.train_cache or (FEATURES_DIR / f"guard_neck_train_{tag}.npz")
    val_cache = args.val_cache or (FEATURES_DIR / f"guard_neck_val_{tag}.npz")
    print("Phase 3 guard-neck fusion v2")
    print("Frozen specialists + frozen guard features -> learned MLP neck.")
    print(f"Device: {device}, train/class={args.examples_per_class}, val/class={args.val_per_class}")
    if args.rebuild_features or not train_cache.exists():
        build_feature_cache(train_cache, args, device, args.examples_per_class, args.seed)
    if args.rebuild_features or not val_cache.exists():
        build_feature_cache(val_cache, args, device, args.val_per_class, args.seed + 101)

    tr = np.load(train_cache, allow_pickle=True)
    va = np.load(val_cache, allow_pickle=True)
    Xtr, ytr = tr["X"].astype(np.float32), tr["y"].astype(np.int64)
    Xv, yv = va["X"].astype(np.float32), va["y"].astype(np.int64)
    train_loader = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)), batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(Xv), torch.from_numpy(yv)), batch_size=args.batch)

    model = GuardNeckFusionHead(GUARD_NECK_INPUT_DIM).to(device)
    criterion = FocalLoss(gamma=2.0, weight=torch.tensor([1.25, 1.0], dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-5)
    best = None
    best_acc = -1.0
    best_loss = float("inf")
    bad = 0
    rows = []
    t0 = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        va_loss, va_acc = run_epoch(model, val_loader, criterion, device)
        scheduler.step(va_loss)
        if va_acc > best_acc or (abs(va_acc - best_acc) < 1e-9 and va_loss < best_loss):
            best_acc, best_loss = va_acc, va_loss
            best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        rows.append({"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc, "val_loss": va_loss, "val_acc": va_acc, "lr": float(optimizer.param_groups[0]["lr"]), "elapsed_sec": time.perf_counter() - t0})
        print(f"Epoch {epoch:02d}/{args.epochs}: train_acc={tr_acc:.3f} val_acc={va_acc:.3f} val_loss={va_loss:.4f}")
        if bad >= 9:
            print("Early stopping.")
            break
    if best is None:
        best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    save_path = args.save_path
    if save_path.exists():
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = save_path.with_name(f"{save_path.stem}_{stamp}{save_path.suffix}")
    metadata = {
        "phase": "phase3_guard_neck_fusion_v2",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "architecture": "Frozen five specialist encoders + Phase2 guard features injected at neck -> MLP head",
        "input_dim": GUARD_NECK_INPUT_DIM,
        "view_names": VIEW_NAMES,
        "guard_feature_names": GUARD_FEATURE_NAMES,
        "classes": ["drone", "no_drone"],
        "drone_idx": 0,
        "train_cache": str(train_cache),
        "val_cache": str(val_cache),
        "best_val_acc": float(best_acc),
        "best_val_loss": float(best_loss),
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
    }
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"phase": metadata["phase"], "head_state_dict": best, "metadata": metadata, "classes": metadata["classes"], "drone_idx": 0}, save_path)
    out_tag = save_path.stem.replace("drone_cnn_", "")
    write_csv(RESULTS_DIR / f"train_log_{out_tag}.csv", rows)
    (RESULTS_DIR / f"metadata_{out_tag}.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved guard-neck fusion model -> {save_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
