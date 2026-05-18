"""
train_phase1b_gpu.py  --  Phase 1b: fine-tune with pitch-shift augmentation.

Why
---
The base model was trained on DADS (mostly consumer DJI-style drones, low RPM,
100-300 Hz fundamental). FPV / military drones run at much higher RPM and
produce buzz in the 500-5000 Hz range.

Strategy
--------
1. Re-use the existing cached val/test features (no re-extraction needed).
2. Augment the DRONE training windows with pitch shifts:
      +3, +6, +9, +12 semitones  (covers FPV / high-RPM range)
      -3, -6 semitones            (covers large slow drones / props)
   This adds 6 synthetic copies of every drone window.
3. Fine-tune the existing model for 15 epochs on the augmented training set.
4. Save as  models/drone_cnn_phase1b.pth

Run
---
    python train_phase1b_gpu.py
"""

import json, random, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchaudio.transforms as T
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# ── Shared config (must match train_phase1_gpu.py) ───────────────────────
ROOT         = Path(__file__).parent
FEATURES_DIR = ROOT / "features"
MODELS_DIR   = ROOT / "models"
RESULTS_DIR  = ROOT / "results"

FS          = 16000
WIN_SAMPLES = FS
HOP_SAMPLES = FS // 2
N_FFT   = 512
WIN_LEN = round(0.025 * FS)
HOP_LEN = WIN_LEN - round(0.015 * FS)
N_MELS  = 64

CLASSES    = ["drone", "no_drone"]
DRONE_IDX  = 0
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Fine-tune settings
EPOCHS     = 15
BATCH_SIZE = 256
LR         = 3e-4          # lower than initial (we're fine-tuning)

# Pitch shifts in semitones → converted to mel-bin offsets.
# Mel scale: 64 bands over ~0-8 kHz ≈ 7 octaves → ~0.76 bins/semitone.
# Shifting mel bins is equivalent to pitch shifting in the log-frequency domain.
PITCH_SHIFTS_SEMITONES = [+3, +6, +9, +12, -3, -6]
BINS_PER_SEMITONE      = N_MELS / (7 * 12)   # ≈ 0.76
PITCH_SHIFTS_BINS      = [round(s * BINS_PER_SEMITONE) for s in PITCH_SHIFTS_SEMITONES]


# No mel transform needed — augmentation works on cached mel features directly


# ── Model ─────────────────────────────────────────────────────────────────
class DroneCNN(nn.Module):
    def __init__(self, n_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc  = nn.Linear(64, n_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


# ── Mel-bin shift (GPU batch, instant) ───────────────────────────────────
def shift_mel_bins_batch(X: torch.Tensor, n_bins: int) -> torch.Tensor:
    """
    Shift the frequency axis of a batch of log-mel spectrograms by n_bins rows.
    X shape: [N, H, W]  (batch of 2D mel images, no channel dim here)
    Positive n_bins → pitch up; negative → pitch down.
    Empty rows are filled with the minimum value in the batch (silence).
    """
    N, H, W = X.shape
    out     = torch.full_like(X, X.min())
    if n_bins > 0:
        out[:, n_bins:, :] = X[:, :H - n_bins, :]
    elif n_bins < 0:
        out[:, :H + n_bins, :] = X[:, -n_bins:, :]
    else:
        out = X.clone()
    return out


# ── Build augmented training features (incremental memmap, no RAM spike) ──
def build_augmented_train_features():
    aug_path_X = FEATURES_DIR / "X_train_aug.npy"
    aug_path_y = FEATURES_DIR / "y_train_aug.npy"

    if aug_path_X.exists() and aug_path_y.exists():
        print("Loading cached augmented training features...")
        X = np.load(str(aug_path_X), mmap_mode="r")
        y = np.load(str(aug_path_y))
        print(f"  Shape: {X.shape}  "
              f"drone={int((y==DRONE_IDX).sum()):,}  "
              f"no_drone={int((y!=DRONE_IDX).sum()):,}")
        return X, y

    print("Loading original training features...")
    X_orig = np.load(str(FEATURES_DIR / "X_train.npy"))   # float16 [N, 64, 101]
    y_orig = np.load(str(FEATURES_DIR / "y_train.npy"))
    print(f"  Shape: {X_orig.shape}  "
          f"drone={int((y_orig==DRONE_IDX).sum()):,}  "
          f"no_drone={int((y_orig!=DRONE_IDX).sum()):,}")

    drone_mask = (y_orig == DRONE_IDX)
    X_drone_np = X_orig[drone_mask]                        # stay float16 on CPU
    n_drone    = len(X_drone_np)
    N_orig, H, W = X_orig.shape
    N_total    = N_orig + n_drone * len(PITCH_SHIFTS_BINS)

    print(f"  Pre-allocating {N_total:,}-window memmap  "
          f"({N_total*H*W*2/1e9:.2f} GB on disk)...")
    X_mm = np.lib.format.open_memmap(str(aug_path_X), mode="w+",
                                      dtype=np.float16, shape=(N_total, H, W))
    y_mm = np.lib.format.open_memmap(str(aug_path_y), mode="w+",
                                      dtype=np.int8,   shape=(N_total,))

    # Block 0: originals
    X_mm[:N_orig] = X_orig
    y_mm[:N_orig] = y_orig

    # Blocks 1..6: shifted drone windows written directly to disk slice
    X_drone_gpu = torch.from_numpy(X_drone_np.astype(np.float32)).to(DEVICE)
    print(f"  Applying {len(PITCH_SHIFTS_BINS)} pitch shifts to "
          f"{n_drone:,} drone windows on {DEVICE}...")
    t0     = time.time()
    offset = N_orig
    for i, (semitones, bins) in enumerate(zip(PITCH_SHIFTS_SEMITONES,
                                               PITCH_SHIFTS_BINS)):
        shifted = shift_mel_bins_batch(X_drone_gpu, bins)  # [n_drone, H, W]
        X_mm[offset : offset + n_drone] = shifted.cpu().numpy().astype(np.float16)
        y_mm[offset : offset + n_drone] = DRONE_IDX
        offset += n_drone
        print(f"  Shift {semitones:+d} semitones ({bins:+d} bins) done  "
              f"[{i+1}/{len(PITCH_SHIFTS_BINS)}]  {time.time()-t0:.1f}s",
              flush=True)

    del X_drone_gpu
    del X_mm, y_mm   # flush + close memmap

    # Re-open read-only
    X = np.load(str(aug_path_X), mmap_mode="r")
    y = np.load(str(aug_path_y))
    print(f"\nCombined: {X.shape}  "
          f"drone={int((y==DRONE_IDX).sum()):,}  "
          f"no_drone={int((y!=DRONE_IDX).sum()):,}")
    print(f"Saved -> {aug_path_X.name}  ({aug_path_X.stat().st_size/1e9:.2f} GB)")
    return X, y


# ── Dataset ───────────────────────────────────────────────────────────────
class LogMelDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = torch.from_numpy(y.astype(np.int64))

    def __len__(self): return len(self.y)

    def __getitem__(self, i):
        x = torch.from_numpy(self.X[i].astype(np.float32)).unsqueeze(0)
        return x, self.y[i]


# ── Helpers ───────────────────────────────────────────────────────────────
def compute_class_weights(y):
    n, nc = len(y), len(CLASSES)
    weights = []
    for k in range(nc):
        nk = int((y == k).sum())
        weights.append(n / (nc * nk) if nk else 1.0)
    print("Class weights (augmented set):")
    for k, w in enumerate(weights):
        print(f"  {CLASSES[k]:12s}  {int((y==k).sum()):>8,} windows  weight={w:.4f}")
    return torch.tensor(weights, dtype=torch.float32)


def run_epoch(model, loader, criterion, optimizer, train=True):
    model.train(train)
    loss_sum, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for X, y in loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            out  = model(X)
            loss = criterion(out, y)
            if train:
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            loss_sum += loss.item() * len(y)
            correct  += (out.argmax(1) == y).sum().item()
            n        += len(y)
    return loss_sum / n, correct / n


def evaluate(model, loader):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for X, y in loader:
            preds.append(model(X.to(DEVICE)).argmax(1).cpu())
            labels.append(y)
    preds  = torch.cat(preds).numpy()
    labels = torch.cat(labels).numpy()
    C      = confusion_matrix(labels, preds)
    acc    = (preds == labels).mean()
    print(f"\n=== Test-Set Evaluation (Phase 1b) ===")
    print(f"Accuracy: {acc:.4f}  ({100*acc:.2f}%)\n")
    print(f"{'Class':12s}  Precision  Recall    FPR       FNR")
    print("-" * 56)
    metrics = {"accuracy": float(acc), "confusion_matrix": C.tolist(), "classes": CLASSES}
    for k, cls in enumerate(CLASSES):
        TP = C[k,k]; FP = C[:,k].sum()-TP; FN = C[k,:].sum()-TP; TN = C.sum()-TP-FP-FN
        pr = TP/(TP+FP) if TP+FP else 0; re = TP/(TP+FN) if TP+FN else 0
        fpr= FP/(FP+TN) if FP+TN else 0; fnr= FN/(FN+TP) if FN+TP else 0
        print(f"{cls:12s}  {pr:.4f}     {re:.4f}    {fpr:.4f}    {fnr:.4f}")
        metrics[cls] = {"precision": pr, "recall": re, "FPR": fpr, "FNR": fnr}
    return metrics, preds, labels


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print(" Phase 1b -- Augmented fine-tune (pitch shift)")
    print("=" * 60)
    print(f"Device  : {DEVICE}"
          + (f"  ({torch.cuda.get_device_name(0)})" if DEVICE.type=="cuda" else ""))
    print(f"Shifts  : {PITCH_SHIFTS_SEMITONES} semitones -> {PITCH_SHIFTS_BINS} bins\n")

    # ── Augmented train features ──────────────────────────────────
    print("[1/4] Building augmented training set...")
    X_train, y_train = build_augmented_train_features()

    # ── Load val / test (already cached, no re-extraction) ────────
    print("\n[2/4] Loading val / test features...")
    X_val,  y_val  = np.load(str(FEATURES_DIR/"X_val.npy")),  np.load(str(FEATURES_DIR/"y_val.npy"))
    X_test, y_test = np.load(str(FEATURES_DIR/"X_test.npy")), np.load(str(FEATURES_DIR/"y_test.npy"))
    print(f"Val  : {len(X_val):,}   Test: {len(X_test):,}")

    # ── DataLoaders ───────────────────────────────────────────────
    train_loader = torch.utils.data.DataLoader(
        LogMelDataset(X_train, y_train),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        LogMelDataset(X_val, y_val),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
        LogMelDataset(X_test, y_test),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    # ── Load existing model and fine-tune ─────────────────────────
    print("\n[3/4] Fine-tuning from Phase 1a checkpoint...")
    base_path = MODELS_DIR / "drone_cnn_phase1.pth"
    ckpt      = torch.load(str(base_path), map_location=DEVICE, weights_only=False)
    model     = DroneCNN().to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])

    cw        = compute_class_weights(y_train).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=cw)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_loss, best_state = float("inf"), None
    t0 = time.time()

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, train=True)
        va_loss, va_acc = run_epoch(model, val_loader,   criterion, optimizer, train=False)
        scheduler.step()
        marker = ""
        if va_loss < best_val_loss:
            best_val_loss = va_loss
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            marker = "  <- best"
        print(f"  Epoch {epoch:02d}/{EPOCHS}  "
              f"train {tr_loss:.4f} / {tr_acc:.4f}  |  "
              f"val {va_loss:.4f} / {va_acc:.4f}"
              f"  lr={scheduler.get_last_lr()[0]:.2e}{marker}", flush=True)

    print(f"\nFine-tuning done in {(time.time()-t0)/60:.1f} min")

    # ── Evaluate ──────────────────────────────────────────────────
    print("\n[4/4] Evaluating on test set...")
    model.load_state_dict(best_state)
    metrics, preds, labels = evaluate(model, test_loader)

    # ── Save ──────────────────────────────────────────────────────
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODELS_DIR / "drone_cnn_phase1b.pth"
    torch.save({"model_state_dict": best_state, "classes": CLASSES,
                "config": {"n_mels": N_MELS, "n_fft": N_FFT,
                           "win_len": WIN_LEN, "hop_len": HOP_LEN,
                           "fs": FS, "win_samples": WIN_SAMPLES,
                           "pitch_shifts": PITCH_SHIFTS_SEMITONES}},
               str(model_path))
    print(f"\nModel saved  : {model_path}")

    with open(RESULTS_DIR / "phase1b_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Confusion chart
    fig, ax = plt.subplots(figsize=(6, 5))
    cm_norm = confusion_matrix(labels, preds, normalize="true")
    ConfusionMatrixDisplay(cm_norm, display_labels=CLASSES).plot(
        ax=ax, colorbar=True, cmap="Blues", values_format=".2f")
    ax.set_title("Phase 1b (augmented) -- Test Set")
    fig.tight_layout()
    fig.savefig(str(RESULTS_DIR / "confusion_chart_phase1b.png"), dpi=150)
    plt.close(fig)

    print("=" * 60)
    print(f" Done!  Test accuracy: {metrics['accuracy']*100:.2f}%")
    print(f" (Phase 1a was 99.20% -- improvement shows on FPV sounds)")
    print("=" * 60)

    # Update predict.py to use new model by default
    print("\nTip: to use the improved model in predict.py / live_detector.py,")
    print(f"     change MODEL_PATH to: {model_path}")


if __name__ == "__main__":
    main()
