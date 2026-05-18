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
from torch.utils.data import DataLoader

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.phase2v5_real_noise.audio_phase2v5 import AudioPreprocessor
from src.phase2v5_real_noise.config_phase2v5 import DRONE_DIR, FSD50K_CANDIDATES_CSV, FSD_LABELS, NODRONE_DIR
from src.phase2v5_real_noise.data_phase2v5 import AudioFileWindowPool, FSD50KWindowPool
from src.phase2v5_real_noise.model_phase2v5 import DroneCNNV5, FocalLoss

from src.phase3_real_noise_specialists.config_phase3_specialists import (
    CHECKPOINT_DIR,
    SNR_LEVELS,
    SPECIALIST_BUNDLE_PATH,
    VIEW_NAMES,
    VIEW_WEIGHTS,
    ensure_dirs,
)
from src.phase3_real_noise_specialists.data_phase3_specialists import RealNoiseSpecialistDataset


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def run_epoch(model, loader, criterion, device, optimizer=None):
    train = optimizer is not None
    model.train(train)
    total_loss = 0.0
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        total_loss += float(loss.item()) * y.size(0)
        correct += int((logits.argmax(1) == y).sum().item())
        total += int(y.size(0))
    return total_loss / max(total, 1), correct / max(total, 1)


def build_pools(args, preproc):
    drone_pool = AudioFileWindowPool(DRONE_DIR, preproc, args.max_drone_files, args.seed)
    nodrone_pool = AudioFileWindowPool(NODRONE_DIR, preproc, args.max_nodrone_files, args.seed + 1) if NODRONE_DIR.exists() else None
    fsd_pool = FSD50KWindowPool(FSD50K_CANDIDATES_CSV, preproc, FSD_LABELS, args.max_fsd_clips_per_label, args.seed + 2)
    return drone_pool, fsd_pool, nodrone_pool


def train_one(view_idx, args, pools, preproc, device):
    view_name = VIEW_NAMES[view_idx]
    drone_pool, fsd_pool, nodrone_pool = pools
    train_ds = RealNoiseSpecialistDataset(
        view_idx,
        preproc,
        drone_pool,
        fsd_pool,
        nodrone_pool,
        args.train_per_class,
        args.snr_levels,
        augment=True,
        seed=args.seed + view_idx * 17,
        positive_mix_prob=args.positive_mix_prob,
        negative_fsd_prob=args.negative_fsd_prob,
    )
    val_ds = RealNoiseSpecialistDataset(
        view_idx,
        preproc,
        drone_pool,
        fsd_pool,
        nodrone_pool,
        args.val_per_class,
        args.snr_levels,
        augment=False,
        seed=args.seed + view_idx * 17 + 100,
        positive_mix_prob=args.val_positive_mix_prob,
        negative_fsd_prob=args.negative_fsd_prob,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=args.num_workers)
    model = DroneCNNV5(n_classes=2).to(device)
    criterion = FocalLoss(
        gamma=2.0,
        weight=torch.tensor([args.drone_loss_weight, args.no_drone_loss_weight], dtype=torch.float32, device=device),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4, min_lr=1e-5)
    best_state = None
    best_acc = -1.0
    best_loss = float("inf")
    bad = 0
    rows = []
    t0 = time.perf_counter()
    print(f"\n=== Training specialist {view_idx}: {view_name} ===")
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        va_loss, va_acc = run_epoch(model, val_loader, criterion, device)
        scheduler.step(va_loss)
        improved = va_acc > best_acc or (abs(va_acc - best_acc) < 1e-9 and va_loss < best_loss)
        if improved:
            best_acc = va_acc
            best_loss = va_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        rows.append({
            "view_idx": view_idx,
            "view_name": view_name,
            "epoch": epoch,
            "train_loss": tr_loss,
            "train_acc": tr_acc,
            "val_loss": va_loss,
            "val_acc": va_acc,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "elapsed_sec": time.perf_counter() - t0,
        })
        print(f"{view_name} epoch {epoch:02d}/{args.epochs}: train_acc={tr_acc:.3f} val_acc={va_acc:.3f} val_loss={va_loss:.4f}")
        if bad >= args.patience:
            print(f"{view_name}: early stopping after {bad} non-improving epochs.")
            break
    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    ckpt_path = CHECKPOINT_DIR / f"best_{view_idx}_{view_name.replace('-', '_')}.pth"
    torch.save({"view_idx": view_idx, "view_name": view_name, "model_state_dict": best_state, "best_val_acc": best_acc, "best_val_loss": best_loss}, ckpt_path)
    write_csv(CHECKPOINT_DIR / f"train_log_{view_idx}_{view_name.replace('-', '_')}.csv", rows)
    return best_state, {"view_idx": view_idx, "view_name": view_name, "best_val_acc": float(best_acc), "best_val_loss": float(best_loss), "checkpoint": str(ckpt_path)}


def parse_args():
    ap = argparse.ArgumentParser(description="Phase 3 five real-noise specialists")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--epochs", type=int, default=24)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--torch-threads", type=int, default=3)
    ap.add_argument("--no-gpu", action="store_true")
    ap.add_argument("--seed", type=int, default=3033)
    ap.add_argument("--train-per-class", type=int, default=6000)
    ap.add_argument("--val-per-class", type=int, default=1200)
    ap.add_argument("--max-drone-files", type=int, default=12000)
    ap.add_argument("--max-nodrone-files", type=int, default=5000)
    ap.add_argument("--max-fsd-clips-per-label", type=int, default=500)
    ap.add_argument("--positive-mix-prob", type=float, default=0.95)
    ap.add_argument("--val-positive-mix-prob", type=float, default=1.0)
    ap.add_argument("--negative-fsd-prob", type=float, default=0.95)
    ap.add_argument("--drone-loss-weight", type=float, default=1.45)
    ap.add_argument("--no-drone-loss-weight", type=float, default=1.0)
    ap.add_argument("--snr-levels", type=str, default="-20,-15,-10,-5,0,5,10")
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--save-path", type=Path, default=SPECIALIST_BUNDLE_PATH)
    return ap.parse_args()


def main():
    args = parse_args()
    if args.quick:
        args.epochs = min(args.epochs, 2)
        args.train_per_class = min(args.train_per_class, 350)
        args.val_per_class = min(args.val_per_class, 120)
        args.max_drone_files = min(args.max_drone_files, 800)
        args.max_nodrone_files = min(args.max_nodrone_files, 500)
        args.max_fsd_clips_per_label = min(args.max_fsd_clips_per_label, 25)
        args.save_path = args.save_path.with_name(args.save_path.stem + "_quick" + args.save_path.suffix)
    args.snr_levels = [int(x.strip()) for x in args.snr_levels.split(",") if x.strip()]
    ensure_dirs()
    torch.set_num_threads(max(1, args.torch_threads))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_gpu else "cpu")
    print("Phase 3 real-noise specialists")
    print("No old models will be modified. Five specialists train sequentially.")
    print(f"Device: {device}, batch={args.batch}, workers={args.num_workers}, torch_threads={args.torch_threads}")
    preproc = AudioPreprocessor(16000)
    pools = build_pools(args, preproc)
    print(f"Pools: drone={len(pools[0].paths)}, fsd={len(pools[1].records)}, nodrone={len(pools[2].paths) if pools[2] else 0}")

    best_states = {}
    metrics = []
    for vi in range(len(VIEW_NAMES)):
        state, metric = train_one(vi, args, pools, preproc, device)
        best_states[vi] = state
        metrics.append(metric)

    save_path = args.save_path
    if save_path.exists():
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = save_path.with_name(f"{save_path.stem}_{stamp}{save_path.suffix}")
    bundle = {
        "phase": "phase3_real_noise_specialist_ensemble",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "view_names": VIEW_NAMES,
        "view_weights": VIEW_WEIGHTS.tolist(),
        "classes": ["drone", "no_drone"],
        "drone_idx": 0,
        "metadata": {"metrics": metrics, "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}},
    }
    for vi, name in enumerate(VIEW_NAMES):
        bundle[f"model_{vi}_{name.replace('-', '_')}"] = best_states[vi]
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, save_path)
    (CHECKPOINT_DIR / "training_summary.json").write_text(json.dumps(bundle["metadata"], indent=2), encoding="utf-8")
    print(f"Saved specialist ensemble -> {save_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

