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
    from src.phase2_harmonic_fusion.config_phase2 import DEFAULT_SAVE_PATH, FEATURES_DIR, RESULTS_DIR, ensure_dirs
    from src.phase2_harmonic_fusion.features_phase2 import build_feature_matrix
    from src.phase2_harmonic_fusion.model_phase2 import HarmonicFusionHead
else:
    from .config_phase2 import DEFAULT_SAVE_PATH, FEATURES_DIR, RESULTS_DIR, ensure_dirs
    from .features_phase2 import build_feature_matrix
    from .model_phase2 import HarmonicFusionHead


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def split_features(X, y, seed: int, val_frac: float):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    rng.shuffle(idx)
    n_val = int(round(len(idx) * val_frac))
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]


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
    ap = argparse.ArgumentParser(description="Train Phase 2 harmonic fusion head")
    ap.add_argument("--backbone", type=Path, default=Path("models/phase2v5_real_noise/drone_cnn_phase2v5c_real_noise_balanced.pth"))
    ap.add_argument("--save-path", type=Path, default=DEFAULT_SAVE_PATH)
    ap.add_argument("--feature-cache", type=Path, default=None)
    ap.add_argument("--rebuild-features", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--examples-per-class", type=int, default=4500)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--no-gpu", action="store_true")
    ap.add_argument("--torch-threads", type=int, default=3)
    ap.add_argument("--max-drone-files", type=int, default=12000)
    ap.add_argument("--max-nodrone-files", type=int, default=5000)
    ap.add_argument("--max-fsd-clips-per-label", type=int, default=500)
    return ap.parse_args()


def main():
    args = parse_args()
    if args.quick:
        args.examples_per_class = min(args.examples_per_class, 600)
        args.epochs = min(args.epochs, 12)
        args.max_drone_files = min(args.max_drone_files, 1200)
        args.max_nodrone_files = min(args.max_nodrone_files, 800)
        args.max_fsd_clips_per_label = min(args.max_fsd_clips_per_label, 35)

    ensure_dirs()
    torch.set_num_threads(max(1, args.torch_threads))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_gpu else "cpu")
    feature_cache = args.feature_cache or (FEATURES_DIR / ("phase2_features_quick.npz" if args.quick else "phase2_features_v1.npz"))

    print("Phase 2 harmonic fusion training")
    print("Backbone is frozen; no CNN retraining.")
    print(f"Device: {device}")
    print(f"Backbone: {args.backbone}")

    if args.rebuild_features or not feature_cache.exists():
        build_feature_matrix(
            args.backbone,
            feature_cache,
            args.examples_per_class,
            args.max_drone_files,
            args.max_nodrone_files,
            args.max_fsd_clips_per_label,
            args.seed,
            device,
        )

    data = np.load(feature_cache, allow_pickle=True)
    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.int64)
    X_train, y_train, X_val, y_val = split_features(X, y, args.seed + 1, 0.20)
    train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)), batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)), batch_size=args.batch, shuffle=False)

    model = HarmonicFusionHead(in_dim=X.shape[1]).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([1.25, 1.0], dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-5)

    best_state = None
    best_val = -1.0
    best_loss = float("inf")
    bad = 0
    rows = []
    t0 = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        va_loss, va_acc = run_epoch(model, val_loader, criterion, device)
        scheduler.step(va_loss)
        improved = va_acc > best_val or (abs(va_acc - best_val) < 1e-9 and va_loss < best_loss)
        if improved:
            best_val = va_acc
            best_loss = va_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        row = {
            "epoch": epoch,
            "train_loss": tr_loss,
            "train_acc": tr_acc,
            "val_loss": va_loss,
            "val_acc": va_acc,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "elapsed_sec": time.perf_counter() - t0,
        }
        rows.append(row)
        print(f"Epoch {epoch:02d}/{args.epochs}: train_acc={tr_acc:.3f} val_acc={va_acc:.3f} val_loss={va_loss:.4f}")
        if bad >= 8:
            print("Early stopping.")
            break

    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    metadata = {
        "phase": "phase2_harmonic_fusion_v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "backbone_path": str(args.backbone),
        "feature_cache": str(feature_cache),
        "classes": ["drone", "no_drone"],
        "drone_idx": 0,
        "input_dim": int(X.shape[1]),
        "cnn_latent_dim": 64,
        "harmonic_dim": int(X.shape[1] - 64),
        "best_val_acc": float(best_val),
        "best_val_loss": float(best_loss),
        "note": "Frozen Phase 2v5 CNN weighted latent + non-destructive harmonic features.",
    }
    ckpt = {
        "phase": metadata["phase"],
        "head_state_dict": best_state,
        "metadata": metadata,
        "classes": metadata["classes"],
        "drone_idx": 0,
    }
    save_path = args.save_path
    if save_path.exists():
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = save_path.with_name(f"{save_path.stem}_{stamp}{save_path.suffix}")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, save_path)
    tag = save_path.stem.replace("drone_cnn_", "")
    write_csv(RESULTS_DIR / f"train_log_{tag}.csv", rows)
    (RESULTS_DIR / f"metadata_{tag}.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved Phase 2 head -> {save_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

