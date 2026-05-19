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
from src.phase2v5_real_noise.model_phase2v5 import FocalLoss
from src.phase3_real_noise_specialists.predict_phase3_hybrid import load_specialist_bundle

from src.phase3_mid_fusion.config_mid_fusion import (
    MID_FUSION_INPUT_DIM,
    PHASE2_BACKBONE_PATH,
    PHASE2_GUARD_PATH,
    RESULTS_DIR,
    SAVE_PATH,
    SNR_LEVELS,
    SPECIALIST_BUNDLE_PATH,
    VIEW_NAMES,
    ensure_dirs,
)
from src.phase3_mid_fusion.data_mid_fusion import RealNoiseMidFusionDataset
from src.phase3_mid_fusion.model_mid_fusion import FrozenSpecialistMidFusion, MidFusionHead


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


def run_epoch(model, loader, criterion, device, optimizer=None):
    train = optimizer is not None
    model.head.train(train)
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
            torch.nn.utils.clip_grad_norm_(model.head.parameters(), 5.0)
            optimizer.step()
        total_loss += float(loss.item()) * y.size(0)
        correct += int((logits.argmax(1) == y).sum().item())
        total += int(y.size(0))
    return total_loss / max(total, 1), correct / max(total, 1)


def parse_args():
    ap = argparse.ArgumentParser(description="Train mid-fusion head over frozen five-view specialists")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--examples-per-class", type=int, default=4000)
    ap.add_argument("--val-per-class", type=int, default=900)
    ap.add_argument("--epochs", type=int, default=28)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=6262)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--torch-threads", type=int, default=3)
    ap.add_argument("--no-gpu", action="store_true")
    ap.add_argument("--max-drone-files", type=int, default=12000)
    ap.add_argument("--max-nodrone-files", type=int, default=5000)
    ap.add_argument("--max-fsd-clips-per-label", type=int, default=500)
    ap.add_argument("--positive-mix-prob", type=float, default=0.95)
    ap.add_argument("--val-positive-mix-prob", type=float, default=1.0)
    ap.add_argument("--negative-fsd-prob", type=float, default=0.95)
    ap.add_argument("--drone-loss-weight", type=float, default=1.30)
    ap.add_argument("--no-drone-loss-weight", type=float, default=1.0)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--specialists", type=Path, default=SPECIALIST_BUNDLE_PATH)
    ap.add_argument("--save-path", type=Path, default=SAVE_PATH)
    return ap.parse_args()


def main():
    args = parse_args()
    if args.quick:
        args.examples_per_class = min(args.examples_per_class, 300)
        args.val_per_class = min(args.val_per_class, 100)
        args.epochs = min(args.epochs, 8)
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
    print("Phase 3 mid-fusion v1")
    print("No old models are modified. Specialist CNNs are frozen; only the fusion head trains.")
    print(f"Device: {device}, batch={args.batch}, examples/class={args.examples_per_class}, epochs={args.epochs}")

    preproc = AudioPreprocessor(16000)
    pools = build_pools(args, preproc)
    specialists = load_specialist_bundle(args.specialists, device)["models"]
    model = FrozenSpecialistMidFusion(specialists, MidFusionHead(MID_FUSION_INPUT_DIM)).to(device)

    train_ds = RealNoiseMidFusionDataset(
        preproc,
        pools[0],
        pools[1],
        pools[2],
        args.examples_per_class,
        SNR_LEVELS,
        augment=True,
        seed=args.seed,
        positive_mix_prob=args.positive_mix_prob,
        negative_fsd_prob=args.negative_fsd_prob,
    )
    val_ds = RealNoiseMidFusionDataset(
        preproc,
        pools[0],
        pools[1],
        pools[2],
        args.val_per_class,
        SNR_LEVELS,
        augment=False,
        seed=args.seed + 101,
        positive_mix_prob=args.val_positive_mix_prob,
        negative_fsd_prob=args.negative_fsd_prob,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=args.num_workers)

    criterion = FocalLoss(
        gamma=2.0,
        weight=torch.tensor([args.drone_loss_weight, args.no_drone_loss_weight], dtype=torch.float32, device=device),
    )
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=args.lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4, min_lr=1e-5)

    best_state = None
    best_acc = -1.0
    best_loss = float("inf")
    bad = 0
    rows = []
    t0 = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        va_loss, va_acc = run_epoch(model, val_loader, criterion, device)
        scheduler.step(va_loss)
        improved = va_acc > best_acc or (abs(va_acc - best_acc) < 1e-9 and va_loss < best_loss)
        if improved:
            best_acc = va_acc
            best_loss = va_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.head.state_dict().items()}
            bad = 0
        else:
            bad += 1
        rows.append({
            "epoch": epoch,
            "train_loss": tr_loss,
            "train_acc": tr_acc,
            "val_loss": va_loss,
            "val_acc": va_acc,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "elapsed_sec": time.perf_counter() - t0,
        })
        print(f"Epoch {epoch:02d}/{args.epochs}: train_acc={tr_acc:.3f} val_acc={va_acc:.3f} val_loss={va_loss:.4f}")
        if bad >= args.patience:
            print("Early stopping.")
            break
    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.head.state_dict().items()}

    save_path = args.save_path
    if save_path.exists():
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = save_path.with_name(f"{save_path.stem}_{stamp}{save_path.suffix}")
    metadata = {
        "phase": "phase3_mid_fusion_v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "architecture": "Frozen five DroneCNNV5 encoders -> concat 5x64 latent -> MLP head",
        "input_dim": MID_FUSION_INPUT_DIM,
        "view_names": VIEW_NAMES,
        "classes": ["drone", "no_drone"],
        "drone_idx": 0,
        "specialist_bundle_path": str(args.specialists),
        "phase2_guard_path_for_comparison": str(PHASE2_GUARD_PATH),
        "phase2_backbone_path_for_comparison": str(PHASE2_BACKBONE_PATH),
        "best_val_acc": float(best_acc),
        "best_val_loss": float(best_loss),
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
    }
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"phase": metadata["phase"], "head_state_dict": best_state, "metadata": metadata, "classes": metadata["classes"], "drone_idx": 0}, save_path)
    tag = save_path.stem.replace("drone_cnn_", "")
    write_csv(RESULTS_DIR / f"train_log_{tag}.csv", rows)
    (RESULTS_DIR / f"metadata_{tag}.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved mid-fusion model -> {save_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
