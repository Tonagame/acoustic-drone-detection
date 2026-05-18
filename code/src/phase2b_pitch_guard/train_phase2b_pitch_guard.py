from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.phase2b_pitch_guard.config_phase2b import FEATURES_DIR, FEATURE_NAMES, RESULTS_DIR, SAVE_PATH, ensure_dirs
from src.phase2b_pitch_guard.features_phase2b import build_feature_matrix
from src.phase2b_pitch_guard.model_phase2b import PitchGuardMLP


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def split(X, y, seed, val_frac=0.2):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    rng.shuffle(idx)
    n_val = int(round(len(idx) * val_frac))
    return X[idx[n_val:]], y[idx[n_val:]], X[idx[:n_val]], y[idx[:n_val]]


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
    ap = argparse.ArgumentParser(description="Train Phase 2b learned pitch guard")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--examples-per-class", type=int, default=2000)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=5252)
    ap.add_argument("--no-gpu", action="store_true")
    ap.add_argument("--torch-threads", type=int, default=3)
    ap.add_argument("--max-drone-files", type=int, default=12000)
    ap.add_argument("--max-nodrone-files", type=int, default=5000)
    ap.add_argument("--max-fsd-clips-per-label", type=int, default=500)
    ap.add_argument("--feature-cache", type=Path, default=None)
    ap.add_argument("--rebuild-features", action="store_true")
    ap.add_argument("--save-path", type=Path, default=SAVE_PATH)
    return ap.parse_args()


def main():
    args = parse_args()
    if args.quick:
        args.examples_per_class = min(args.examples_per_class, 350)
        args.epochs = min(args.epochs, 12)
        args.max_drone_files = min(args.max_drone_files, 1000)
        args.max_nodrone_files = min(args.max_nodrone_files, 700)
        args.max_fsd_clips_per_label = min(args.max_fsd_clips_per_label, 35)
        args.save_path = args.save_path.with_name(args.save_path.stem + "_quick" + args.save_path.suffix)
    ensure_dirs()
    torch.set_num_threads(max(1, args.torch_threads))
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_gpu else "cpu")
    feature_cache = args.feature_cache or (FEATURES_DIR / ("phase2b_pitch_features_quick.npz" if args.quick else "phase2b_pitch_features_v1.npz"))
    print("Phase 2b learned pitch guard")
    print(f"Device: {device}")
    if args.rebuild_features or not feature_cache.exists():
        build_feature_matrix(
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
    Xtr, ytr, Xv, yv = split(X, y, args.seed + 1)
    train_loader = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)), batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(Xv), torch.from_numpy(yv)), batch_size=args.batch)
    model = PitchGuardMLP(in_dim=X.shape[1]).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([1.20, 1.0], dtype=torch.float32, device=device))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=5, min_lr=1e-5)
    best = None
    best_acc = -1.0
    best_loss = float("inf")
    bad = 0
    rows = []
    t0 = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, device, opt)
        va_loss, va_acc = run_epoch(model, val_loader, criterion, device)
        sched.step(va_loss)
        if va_acc > best_acc or (abs(va_acc - best_acc) < 1e-9 and va_loss < best_loss):
            best_acc, best_loss = va_acc, va_loss
            best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        rows.append({"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc, "val_loss": va_loss, "val_acc": va_acc, "elapsed_sec": time.perf_counter() - t0})
        print(f"Epoch {epoch:02d}/{args.epochs}: train_acc={tr_acc:.3f} val_acc={va_acc:.3f} val_loss={va_loss:.4f}")
        if bad >= 8:
            print("Early stopping.")
            break
    if best is None:
        best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    save_path = args.save_path
    if save_path.exists():
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = save_path.with_name(f"{save_path.stem}_{stamp}{save_path.suffix}")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "phase": "phase2b_learned_pitch_guard",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "feature_cache": str(feature_cache),
        "input_dim": int(X.shape[1]),
        "feature_names": FEATURE_NAMES,
        "classes": ["drone", "no_drone"],
        "drone_idx": 0,
        "best_val_acc": float(best_acc),
        "best_val_loss": float(best_loss),
        "note": "MLP over Phase3 specialist scores, Phase2 harmonic guard features, and CREPE pitch estimator features.",
    }
    torch.save({"phase": metadata["phase"], "model_state_dict": best, "metadata": metadata, "classes": metadata["classes"], "drone_idx": 0}, save_path)
    tag = save_path.stem.replace("drone_cnn_", "")
    write_csv(RESULTS_DIR / f"train_log_{tag}.csv", rows)
    (RESULTS_DIR / f"metadata_{tag}.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved Phase 2b pitch guard -> {save_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

