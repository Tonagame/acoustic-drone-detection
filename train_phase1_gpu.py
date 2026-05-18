"""
train_phase1_gpu.py
Phase 1 drone audio classifier — PyTorch / RTX 3070 edition.

Pipeline
--------
1. Scan data/raw/drone  and  data/raw/no_drone
2. Split by FILE (70 / 15 / 15) to avoid data leakage
3. Extract log-Mel spectrograms and cache to  features/  (float16, one-time)
4. Train a small CNN with class-weighted cross-entropy on GPU
5. Evaluate on the held-out test set; save model + metrics + confusion chart

Run from the project root:
    python train_phase1_gpu.py
"""

import os, json, time, random
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torchaudio.transforms as T
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# ═══════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════
ROOT         = Path(__file__).parent
DATA_RAW     = ROOT / "data" / "raw"
FEATURES_DIR = ROOT / "features"
MODELS_DIR   = ROOT / "models"
RESULTS_DIR  = ROOT / "results"

FS          = 16000
WIN_SAMPLES = FS               # 1-second window
HOP_SAMPLES = FS // 2          # 50 % overlap

# Log-Mel parameters  (matches MATLAB pipeline)
N_FFT       = 512
WIN_LEN     = round(0.025 * FS)   # 400 samples
HOP_LEN     = WIN_LEN - round(0.015 * FS)  # 400-240 = 160 samples
N_MELS      = 64

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
SEED        = 42

EPOCHS      = 30
BATCH_SIZE  = 256
LR          = 1e-3
LR_DROP_EVERY = 10   # halve LR every N epochs

CLASSES     = ["drone", "no_drone"]   # alphabetical = class indices 0, 1
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ═══════════════════════════════════════════════════════════════════
# 1.  Scan files
# ═══════════════════════════════════════════════════════════════════
def scan_files():
    files = []
    for label_idx, class_name in enumerate(CLASSES):
        folder = DATA_RAW / class_name
        if not folder.is_dir():
            raise FileNotFoundError(f"Folder not found: {folder}")
        wavs = sorted(folder.glob("*.wav"))
        if not wavs:
            raise FileNotFoundError(f"No WAV files in {folder}")
        for p in wavs:
            files.append((p, label_idx))
    print(f"Total files: {len(files)}  "
          f"({sum(1 for _,l in files if l==0)} drone, "
          f"{sum(1 for _,l in files if l==1)} no_drone)")
    return files


# ═══════════════════════════════════════════════════════════════════
# 2.  Split by file
# ═══════════════════════════════════════════════════════════════════
def split_files(files):
    random.seed(SEED)
    idx = list(range(len(files)))
    random.shuffle(idx)
    n_train = round(TRAIN_RATIO * len(files))
    n_val   = round(VAL_RATIO   * len(files))
    train = [files[i] for i in idx[:n_train]]
    val   = [files[i] for i in idx[n_train : n_train + n_val]]
    test  = [files[i] for i in idx[n_train + n_val:]]
    print(f"File split  ->  train: {len(train)} | val: {len(val)} | test: {len(test)}")
    return train, val, test


# ═══════════════════════════════════════════════════════════════════
# 3.  Feature extraction + caching
# ═══════════════════════════════════════════════════════════════════
_mel_transform = T.MelSpectrogram(
    sample_rate   = FS,
    n_fft         = N_FFT,
    win_length    = WIN_LEN,
    hop_length    = HOP_LEN,
    n_mels        = N_MELS,
    power         = 2.0,
)

def audio_to_logmel(audio: np.ndarray) -> np.ndarray:
    """Return log-Mel spectrogram [N_MELS x T] for a 1-second mono window."""
    t = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0)  # [1, 16000]
    mel = _mel_transform(t)                                        # [1, N_MELS, T]
    log_mel = torch.log10(mel + 1e-10).squeeze(0).numpy()         # [N_MELS, T]
    return log_mel


def extract_windows(audio: np.ndarray, sr: int) -> list:
    """Resample, mono, normalise, then slice into 1-s windows."""
    # Mono
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    # Resample
    if sr != FS:
        import torchaudio.functional as F
        t = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0)
        t = F.resample(t, sr, FS)
        audio = t.squeeze(0).numpy()
    # Normalise
    peak = np.abs(audio).max()
    if peak > 0:
        audio = audio / peak
    # Slice
    windows = []
    starts = range(0, len(audio) - WIN_SAMPLES + 1, HOP_SAMPLES)
    for s in starts:
        windows.append(audio[s : s + WIN_SAMPLES])
    return windows


def extract_split(split_files, split_name):
    """Extract features for one split; return (X float16, y int8)."""
    X_list, y_list = [], []
    total = len(split_files)
    t0 = time.time()
    for k, (path, label) in enumerate(split_files):
        try:
            audio, sr = sf.read(str(path), dtype="float32")
        except Exception as e:
            print(f"  [WARN] {path.name}: {e}")
            continue

        for win in extract_windows(audio, sr):
            logmel = audio_to_logmel(win)      # [64, T]
            X_list.append(logmel.astype(np.float16))
            y_list.append(label)

        if (k + 1) % 5000 == 0 or (k + 1) == total:
            elapsed = time.time() - t0
            pct = 100 * (k + 1) / total
            print(f"  [{split_name}] {k+1:>6}/{total}  ({pct:.0f}%)  "
                  f"{elapsed:.0f}s  windows so far: {len(X_list):,}", flush=True)

    X = np.stack(X_list)    # [N, 64, T]
    y = np.array(y_list, dtype=np.int8)
    return X, y


def load_or_extract_features(train_files, val_files, test_files):
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    splits = {"train": train_files, "val": val_files, "test": test_files}
    data = {}

    for name, file_list in splits.items():
        x_path = FEATURES_DIR / f"X_{name}.npy"
        y_path = FEATURES_DIR / f"y_{name}.npy"

        if x_path.exists() and y_path.exists():
            print(f"  Loading cached {name} features from {x_path.name} ...")
            X = np.load(str(x_path))
            y = np.load(str(y_path))
        else:
            print(f"\nExtracting {name} features ({len(file_list)} files)...")
            X, y = extract_split(file_list, name)
            np.save(str(x_path), X)
            np.save(str(y_path), y)
            print(f"  Saved {name}: X={X.shape}  y={y.shape}")

        data[name] = (X, y)
        print(f"  {name:5s}: {X.shape[0]:,} windows  "
              f"(drone={int((y==0).sum()):,}  no_drone={int((y==1).sum()):,})")

    return data


# ═══════════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════════
class LogMelDataset(torch.utils.data.Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        # X: [N, 64, T] float16  →  store as-is, cast in __getitem__
        self.X = X
        self.y = torch.from_numpy(y.astype(np.int64))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        x = torch.from_numpy(self.X[i].astype(np.float32)).unsqueeze(0)  # [1, 64, T]
        return x, self.y[i]


# ═══════════════════════════════════════════════════════════════════
# Model  (matches MATLAB CNN)
# ═══════════════════════════════════════════════════════════════════
class DroneCNN(nn.Module):
    def __init__(self, n_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # Block 2
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # Block 3
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc  = nn.Linear(64, n_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


# ═══════════════════════════════════════════════════════════════════
# Training helpers
# ═══════════════════════════════════════════════════════════════════
def compute_class_weights(y: np.ndarray) -> torch.Tensor:
    n_total  = len(y)
    n_cls    = len(CLASSES)
    weights  = []
    print("Class weights:")
    for k in range(n_cls):
        nk = int((y == k).sum())
        w  = n_total / (n_cls * nk) if nk > 0 else 1.0
        weights.append(w)
        print(f"  {CLASSES[k]:12s}  {nk:>8,} samples  weight = {w:.4f}")
    return torch.tensor(weights, dtype=torch.float32)


def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train(train)
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            loss   = criterion(logits, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(y)
            correct    += (logits.argmax(1) == y).sum().item()
            n          += len(y)
    return total_loss / n, correct / n


# ═══════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════
def evaluate_model(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X, y in loader:
            preds = model(X.to(device)).argmax(1).cpu()
            all_preds.append(preds)
            all_labels.append(y)
    preds  = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()

    C        = confusion_matrix(labels, preds)
    accuracy = (preds == labels).mean()
    metrics  = {"accuracy": float(accuracy), "classes": CLASSES, "confusion_matrix": C.tolist()}

    print(f"\n=== Test-Set Evaluation ===")
    print(f"Accuracy: {accuracy:.4f}  ({100*accuracy:.2f}%)\n")
    print(f"{'Class':12s}  Precision  Recall    FPR       FNR")
    print("-" * 56)
    for k, cls in enumerate(CLASSES):
        TP = C[k, k]
        FP = C[:, k].sum() - TP
        FN = C[k, :].sum() - TP
        TN = C.sum()       - TP - FP - FN
        prec = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        rec  = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        fpr  = FP / (FP + TN) if (FP + TN) > 0 else 0.0
        fnr  = FN / (FN + TP) if (FN + TP) > 0 else 0.0
        metrics[cls] = {"precision": prec, "recall": rec, "FPR": fpr, "FNR": fnr}
        print(f"{cls:12s}  {prec:.4f}     {rec:.4f}    {fpr:.4f}    {fnr:.4f}")

    return metrics, preds, labels


def save_confusion_chart(labels, preds, out_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    cm_norm = confusion_matrix(labels, preds, normalize="true")
    disp = ConfusionMatrixDisplay(cm_norm, display_labels=CLASSES)
    disp.plot(ax=ax, colorbar=True, cmap="Blues", values_format=".2f")
    ax.set_title("Drone vs No-Drone  –  Test Set")
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)
    print(f"Confusion chart saved: {out_path}")


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print(" Phase 1 — Drone Audio Detector  (PyTorch / GPU)")
    print("=" * 60)
    print(f"Device : {DEVICE}"
          + (f"  ({torch.cuda.get_device_name(0)})" if DEVICE.type == "cuda" else ""))
    print(f"Root   : {ROOT}\n")

    # ── 1. Scan + split ───────────────────────────────────────────
    files = scan_files()
    train_files, val_files, test_files = split_files(files)

    # ── 2. Extract / load features ────────────────────────────────
    print("\n[Feature extraction]")
    data = load_or_extract_features(train_files, val_files, test_files)
    X_train, y_train = data["train"]
    X_val,   y_val   = data["val"]
    X_test,  y_test  = data["test"]

    # ── 3. DataLoaders ────────────────────────────────────────────
    # num_workers=0 avoids Windows multiprocessing issues
    train_loader = torch.utils.data.DataLoader(
        LogMelDataset(X_train, y_train),
        batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        LogMelDataset(X_val,   y_val),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
        LogMelDataset(X_test,  y_test),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    # ── 4. Model + loss + optimiser ───────────────────────────────
    model     = DroneCNN().to(DEVICE)
    cw        = compute_class_weights(y_train).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=cw)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=LR_DROP_EVERY, gamma=0.5)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters: {total_params:,}")
    print(f"Train windows   : {len(X_train):,}")

    # ── 5. Training loop ──────────────────────────────────────────
    print(f"\n[Training — {EPOCHS} epochs, batch {BATCH_SIZE}]")
    best_val_loss = float("inf")
    best_state    = None
    history       = []
    t_start       = time.time()

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, DEVICE, train=True)
        va_loss, va_acc = run_epoch(model, val_loader,   criterion, optimizer, DEVICE, train=False)
        scheduler.step()

        history.append({"epoch": epoch, "tr_loss": tr_loss, "tr_acc": tr_acc,
                         "va_loss": va_loss, "va_acc": va_acc})

        marker = ""
        if va_loss < best_val_loss:
            best_val_loss = va_loss
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            marker = "  <- best"

        print(f"  Epoch {epoch:02d}/{EPOCHS}  "
              f"train loss {tr_loss:.4f} acc {tr_acc:.4f}  |  "
              f"val loss {va_loss:.4f} acc {va_acc:.4f}"
              f"  lr={scheduler.get_last_lr()[0]:.2e}{marker}", flush=True)

    elapsed = time.time() - t_start
    print(f"\nTraining complete in {elapsed/60:.1f} min")

    # ── 6. Load best weights + evaluate ──────────────────────────
    model.load_state_dict(best_state)
    metrics, preds, labels = evaluate_model(model, test_loader, DEVICE)

    # ── 7. Save model ─────────────────────────────────────────────
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "drone_cnn_phase1.pth"
    torch.save({"model_state_dict": best_state,
                "classes": CLASSES,
                "config": {"n_mels": N_MELS, "n_fft": N_FFT,
                           "win_len": WIN_LEN, "hop_len": HOP_LEN,
                           "fs": FS, "win_samples": WIN_SAMPLES}},
               str(model_path))
    print(f"\nModel saved: {model_path}")

    # ── 8. Save results ───────────────────────────────────────────
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    metrics["history"] = history
    with open(RESULTS_DIR / "phase1_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    save_confusion_chart(labels, preds, RESULTS_DIR / "confusion_chart.png")

    print("\n" + "=" * 60)
    print(f" Done!  Test accuracy: {metrics['accuracy']*100:.2f}%")
    print(f" Model  : {model_path}")
    print(f" Results: {RESULTS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
