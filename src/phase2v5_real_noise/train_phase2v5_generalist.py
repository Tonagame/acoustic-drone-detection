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
    from src.phase2v5_real_noise.config_phase2v5 import (
        DEFAULT_SAVE_PATH,
        LATENTS_DIR,
        QUICK_SAVE_PATH,
        RESULTS_DIR,
        SNR_LEVELS,
        VIEW_NAMES,
        VIEW_WEIGHTS,
        ensure_dirs,
    )
    from src.phase2v5_real_noise.data_phase2v5 import RealNoiseGeneralistDataset, build_pools
    from src.phase2v5_real_noise.model_phase2v5 import DroneCNNV5, FocalLoss
else:
    from .audio_phase2v5 import AudioPreprocessor
    from .config_phase2v5 import (
        DEFAULT_SAVE_PATH,
        LATENTS_DIR,
        QUICK_SAVE_PATH,
        RESULTS_DIR,
        SNR_LEVELS,
        VIEW_NAMES,
        VIEW_WEIGHTS,
        ensure_dirs,
    )
    from .data_phase2v5 import RealNoiseGeneralistDataset, build_pools
    from .model_phase2v5 import DroneCNNV5, FocalLoss


def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, criterion, device, optimizer=None):
    train = optimizer is not None
    model.train(train)
    total_loss = 0.0
    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device, non_blocking=False)
        y = y.to(device, non_blocking=False)
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


@torch.no_grad()
def save_clean_drone_latents(model, drone_pool, preproc, device, out_path: Path, max_windows: int, seed: int):
    rng = random.Random(seed)
    model.eval()
    latents = []
    for _ in range(max_windows):
        audio = drone_pool.sample_window(rng)
        view = preproc.create_audio_views(audio)[0]
        lm = preproc.audio_to_logmel(view).unsqueeze(0).unsqueeze(0).to(device)
        latents.append(model.encode(lm).squeeze(0).cpu().numpy())
    arr = np.stack(latents, axis=0).astype(np.float32) if latents else np.zeros((0, 64), dtype=np.float32)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, latents=arr, label=np.zeros(len(arr), dtype=np.int64), note="clean_drone_raw_view")


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def output_tag(args, save_path: Path) -> str:
    if args.quick:
        return "quick"
    if args.save_path is not None:
        return save_path.stem.replace("drone_cnn_", "")
    return "full"


def parse_args():
    ap = argparse.ArgumentParser(description="Phase 2v5 real-noise generalist training")
    ap.add_argument("--quick", action="store_true", help="Gentle pipeline test: 3 epochs and small data caps")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--sample_rate", type=int, default=16000)
    ap.add_argument("--no-gpu", action="store_true")
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--torch-threads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--train-per-class", type=int, default=6000)
    ap.add_argument("--val-per-class", type=int, default=1200)
    ap.add_argument("--max-drone-files", type=int, default=12000)
    ap.add_argument("--max-nodrone-files", type=int, default=5000)
    ap.add_argument("--max-fsd-clips-per-label", type=int, default=350)
    ap.add_argument("--latent-windows", type=int, default=1000)
    ap.add_argument("--positive-mix-prob", type=float, default=0.85)
    ap.add_argument("--val-positive-mix-prob", type=float, default=0.85)
    ap.add_argument("--negative-fsd-prob", type=float, default=0.80)
    ap.add_argument("--drone-loss-weight", type=float, default=1.0)
    ap.add_argument("--no-drone-loss-weight", type=float, default=1.0)
    ap.add_argument("--snr-levels", type=str, default=",".join(str(x) for x in SNR_LEVELS))
    ap.add_argument("--save-path", type=Path, default=None)
    return ap.parse_args()


def main():
    args = parse_args()
    if args.quick:
        args.epochs = min(args.epochs, 3)
        args.train_per_class = min(args.train_per_class, 600)
        args.val_per_class = min(args.val_per_class, 200)
        args.max_drone_files = min(args.max_drone_files, 1200)
        args.max_nodrone_files = min(args.max_nodrone_files, 800)
        args.max_fsd_clips_per_label = min(args.max_fsd_clips_per_label, 35)
        args.latent_windows = min(args.latent_windows, 160)

    ensure_dirs()
    snr_levels = [int(x.strip()) for x in args.snr_levels.split(",") if x.strip()]
    torch.set_num_threads(max(1, int(args.torch_threads)))
    set_seeds(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_gpu else "cpu")
    save_path = args.save_path or (QUICK_SAVE_PATH if args.quick else DEFAULT_SAVE_PATH)
    if save_path.exists():
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = save_path.with_name(f"{save_path.stem}_{stamp}{save_path.suffix}")

    print("Phase 2v5 real-noise generalist")
    print("No old files/models will be modified.")
    print(f"Device: {device}, torch_threads={args.torch_threads}, num_workers={args.num_workers}")
    print(f"Checkpoint: {save_path}")

    preproc = AudioPreprocessor(args.sample_rate)
    drone_pool, fsd_pool, nodrone_pool = build_pools(args, preproc)
    print(f"Pools: drone_files={len(drone_pool.paths)}, fsd_clips={len(fsd_pool.records)}, nodrone_files={len(nodrone_pool.paths) if nodrone_pool else 0}")

    train_ds = RealNoiseGeneralistDataset(
        preproc,
        drone_pool,
        fsd_pool,
        nodrone_pool,
        args.train_per_class,
        snr_levels,
        augment=True,
        seed=args.seed,
        positive_mix_prob=args.positive_mix_prob,
        negative_fsd_prob=args.negative_fsd_prob,
        mix_positives=True,
    )
    val_ds = RealNoiseGeneralistDataset(
        preproc,
        drone_pool,
        fsd_pool,
        nodrone_pool,
        args.val_per_class,
        snr_levels,
        augment=False,
        seed=args.seed + 10,
        positive_mix_prob=args.val_positive_mix_prob,
        negative_fsd_prob=args.negative_fsd_prob,
        mix_positives=True,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=args.num_workers)

    model = DroneCNNV5(n_classes=2).to(device)
    weight = torch.tensor([args.drone_loss_weight, args.no_drone_loss_weight], dtype=torch.float32, device=device)
    criterion = FocalLoss(gamma=2.0, weight=weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-5)

    best_state = None
    best_val_acc = -1.0
    best_val_loss = float("inf")
    bad_epochs = 0
    patience = 6
    rows = []
    t0 = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        va_loss, va_acc = run_epoch(model, val_loader, criterion, device)
        scheduler.step(va_loss)
        lr = float(optimizer.param_groups[0]["lr"])
        improved = va_acc > best_val_acc or (abs(va_acc - best_val_acc) < 1e-9 and va_loss < best_val_loss)
        if improved:
            best_val_acc = va_acc
            best_val_loss = va_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
        row = {
            "epoch": epoch,
            "train_loss": tr_loss,
            "train_acc": tr_acc,
            "val_loss": va_loss,
            "val_acc": va_acc,
            "lr": lr,
            "elapsed_sec": time.perf_counter() - t0,
        }
        rows.append(row)
        print(f"Epoch {epoch:02d}/{args.epochs}: train_acc={tr_acc:.3f} val_acc={va_acc:.3f} val_loss={va_loss:.4f} lr={lr:.2e}")
        if bad_epochs >= patience:
            print(f"Early stopping after {bad_epochs} non-improving epochs.")
            break

    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)

    serializable_args = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    serializable_args["save_path"] = str(save_path)
    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "phase": "phase2v5_real_noise_generalist",
        "quick": bool(args.quick),
        "sample_rate": int(args.sample_rate),
        "classes": ["drone", "no_drone"],
        "drone_idx": 0,
        "view_names": VIEW_NAMES,
        "view_weights": VIEW_WEIGHTS.tolist(),
        "architecture": "DroneCNNV5",
        "data_recipe": "DADS drone + FSD50K vehicle/engine hard negatives + DADS/FSD50K mixed positives",
        "snr_levels": snr_levels,
        "args": serializable_args,
        "best_val_acc": float(best_val_acc),
        "best_val_loss": float(best_val_loss),
    }
    ckpt = {
        "phase": metadata["phase"],
        "model_state_dict": best_state,
        "classes": metadata["classes"],
        "drone_idx": 0,
        "sample_rate": int(args.sample_rate),
        "view_names": VIEW_NAMES,
        "view_weights": VIEW_WEIGHTS.tolist(),
        "metadata": metadata,
    }
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, save_path)
    tag = output_tag(args, save_path)
    log_path = RESULTS_DIR / f"phase1_train_log_{tag}.csv"
    write_csv(log_path, rows)
    (RESULTS_DIR / f"phase1_metadata_{tag}.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    latent_path = LATENTS_DIR / f"clean_drone_latents_{tag}.npz"
    save_clean_drone_latents(model, drone_pool, preproc, device, latent_path, args.latent_windows, args.seed + 99)
    print(f"Saved checkpoint -> {save_path}")
    print(f"Saved train log -> {log_path}")
    print(f"Saved clean-drone latents -> {latent_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
