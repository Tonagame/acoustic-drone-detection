"""
train_phase2v2_gpu.py  --  Phase 2 v2: speech-robust + noise-robust

Why
---
Phase 1b detects drones well in quiet conditions and covers FPV frequencies.
However it was never trained on human speech, so someone talking near the
microphone can trigger a false alarm.  This phase fixes that.

What's new
----------
1. Downloads LibriSpeech test-clean (~346 MB) and adds speech windows as
   no_drone training examples.
2. Applies online SpecAugment (random freq + time masking) during training
   to make features more noise-robust without extra disk storage.
3. Fine-tunes from drone_cnn_phase1b.pth (falls back to phase1 if absent).
4. Evaluates:
     • Overall accuracy + confusion matrix
     • Speech false alarm rate (FAR) on held-out speech
     • SNR robustness curve (Gaussian noise on test features)

Memory-safe design
------------------
• Base 4 GB aug cache loaded with mmap (reads from disk lazily).
• Speech features extracted in GPU batches of 64, cached as float16.
• DataLoader num_workers=0 (Windows-safe).
• No giant in-RAM concatenations.

Run
---
    python train_phase2v2_gpu.py

Output
------
    models/drone_cnn_phase2v2.pth
    results/phase2v2/metrics.json
    results/phase2v2/confusion_chart.png
    results/phase2v2/snr_curve.png
"""

import json, random, ssl, time, tarfile, urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
import torchaudio.functional as FA
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# ── Config ────────────────────────────────────────────────────────────────
ROOT         = Path(__file__).parent
FEATURES_DIR = ROOT / "features"
MODELS_DIR   = ROOT / "models"
RESULTS_DIR  = ROOT / "results" / "phase2v2"
DATA_DIR     = ROOT / "data" / "raw"
SPEECH_DIR   = DATA_DIR / "speech" / "librispeech"

FS          = 16000
WIN_SAMPLES = FS
HOP_SAMPLES = FS // 2
N_FFT   = 512
WIN_LEN = round(0.025 * FS)   # 400 samples = 25 ms
HOP_LEN = WIN_LEN - round(0.015 * FS)  # 160 samples = 10 ms
N_MELS  = 64

CLASSES   = ["drone", "no_drone"]
DRONE_IDX = 0
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Fine-tune settings
EPOCHS     = 20
BATCH_SIZE = 512
LR         = 1e-4

# SpecAugment (applied online during training)
FREQ_MASK_PARAM = 8    # max mel bands to mask
TIME_MASK_PARAM = 20   # max time frames to mask
N_FREQ_MASKS    = 2    # number of frequency masks per sample
N_TIME_MASKS    = 2    # number of time masks per sample
SPEC_AUG_PROB   = 0.5  # probability of augmenting each sample

# SNR evaluation
SNR_LEVELS_DB = [-15, -10, -5, 0, 5, 10]

# LibriSpeech
LIBRISPEECH_URL = "https://us.openslr.org/resources/12/test-clean.tar.gz"
SPEECH_TRAIN_SPLIT = 0.8   # 80% → training no_drone  /  20% → FAR evaluation


# ── Mel transform (GPU) ───────────────────────────────────────────────────
_mel = T.MelSpectrogram(
    sample_rate=FS, n_fft=N_FFT,
    win_length=WIN_LEN, hop_length=HOP_LEN,
    n_mels=N_MELS, power=2.0,
).to(DEVICE)


@torch.no_grad()
def batch_wav_to_logmel(wav_np: np.ndarray) -> np.ndarray:
    """
    wav_np : float32 [N, 16000] numpy array
    returns: float16 [N, 64, 101] log-mel on CPU
    """
    t = torch.from_numpy(wav_np).to(DEVICE)
    m = _mel(t)
    return torch.log10(m + 1e-10).cpu().numpy().astype(np.float16)


# ── DroneCNN (identical to Phase 1) ───────────────────────────────────────
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


# ── Download LibriSpeech test-clean ───────────────────────────────────────
def download_librispeech():
    SPEECH_DIR.mkdir(parents=True, exist_ok=True)
    existing = list(SPEECH_DIR.rglob("*.flac"))
    if existing:
        print(f"  LibriSpeech already present: {len(existing):,} FLAC files")
        return

    tar_path = SPEECH_DIR / "test-clean.tar.gz"
    if not tar_path.exists():
        print(f"  Downloading LibriSpeech test-clean (~346 MB)...")
        # Bypass SSL verification (same fix as Phase 1 HuggingFace download)
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode    = ssl.CERT_NONE
        opener = urllib.request.build_opener(
            urllib.request.HTTPSHandler(context=ctx))
        urllib.request.install_opener(opener)

        def _progress(count, block, total):
            if total > 0 and count % 1000 == 0:
                pct = min(100, count * block * 100 // total)
                print(f"    {pct:3d}%", end="\r", flush=True)

        urllib.request.urlretrieve(LIBRISPEECH_URL, str(tar_path), _progress)
        print()

    print("  Extracting archive...")
    with tarfile.open(str(tar_path), "r:gz") as tf:
        tf.extractall(str(SPEECH_DIR))
    flacs = list(SPEECH_DIR.rglob("*.flac"))
    print(f"  Extracted {len(flacs):,} FLAC files")


# ── Extract speech features (GPU batch, float16 cache) ────────────────────
def extract_speech_features():
    cache_X_tr = FEATURES_DIR / "X_speech_train.npy"
    cache_y_tr = FEATURES_DIR / "y_speech_train.npy"
    cache_X_te = FEATURES_DIR / "X_speech_test.npy"

    if cache_X_tr.exists() and cache_X_te.exists():
        print("  Loading cached speech features...")
        X_tr = np.load(str(cache_X_tr), mmap_mode="r")
        y_tr = np.load(str(cache_y_tr))
        X_te = np.load(str(cache_X_te), mmap_mode="r")
        print(f"  Train: {len(X_tr):,}  FAR-test: {len(X_te):,} speech windows")
        return X_tr, y_tr, X_te

    flacs = sorted(SPEECH_DIR.rglob("*.flac"))
    if not flacs:
        print("  WARNING: No speech files — skipping speech augmentation.")
        return None, None, None

    print(f"  Extracting {len(flacs):,} FLAC files on {DEVICE} (batches of 64)...")
    t0 = time.time()

    GPU_BATCH = 64
    wav_buf   = []   # accumulate windows before GPU batch
    all_parts = []   # list of float16 arrays

    def _flush():
        if wav_buf:
            arr = np.stack(wav_buf).astype(np.float32)   # [B, 16000]
            all_parts.append(batch_wav_to_logmel(arr))    # [B, 64, 101] float16
            wav_buf.clear()

    for fi, fp in enumerate(flacs):
        try:
            wav, sr = torchaudio.load(str(fp))
            wav = wav.mean(0).numpy()                   # stereo→mono, CPU numpy
            if sr != FS:
                # resample on CPU for small clips
                t_wav = torch.from_numpy(wav).unsqueeze(0)
                wav   = FA.resample(t_wav, sr, FS).squeeze(0).numpy()
            peak = np.abs(wav).max()
            if peak < 1e-4:
                continue
            wav = wav / peak
            # Slice into 1-second windows with 50 % overlap
            for s in range(0, len(wav) - WIN_SAMPLES + 1, HOP_SAMPLES):
                wav_buf.append(wav[s : s + WIN_SAMPLES])
                if len(wav_buf) >= GPU_BATCH:
                    _flush()
        except Exception:
            continue

        if (fi + 1) % 500 == 0:
            total_so_far = sum(len(p) for p in all_parts) + len(wav_buf)
            print(f"    {fi+1:,}/{len(flacs):,}  windows: {total_so_far:,}  "
                  f"{time.time()-t0:.0f}s", flush=True)

    _flush()   # remaining

    all_windows = np.concatenate(all_parts, axis=0)   # [N, 64, 101] float16
    print(f"  Total speech windows: {len(all_windows):,}  ({time.time()-t0:.1f}s)")

    # Shuffle and split 80 / 20
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(all_windows))
    n_tr = int(len(all_windows) * SPEECH_TRAIN_SPLIT)
    X_tr = all_windows[idx[:n_tr]]
    X_te = all_windows[idx[n_tr:]]
    y_tr = np.ones(n_tr, dtype=np.int8)   # label 1 = no_drone

    np.save(str(cache_X_tr), X_tr)
    np.save(str(cache_y_tr), y_tr)
    np.save(str(cache_X_te), X_te)
    print(f"  Cached: {n_tr:,} train  {len(X_te):,} FAR-test speech windows")

    return (np.load(str(cache_X_tr), mmap_mode="r"),
            np.load(str(cache_y_tr)),
            np.load(str(cache_X_te), mmap_mode="r"))


# ── SpecAugment (CPU, applied per sample in __getitem__) ──────────────────
def spec_augment(x: torch.Tensor) -> torch.Tensor:
    """
    x : [1, H, W]  log-mel tensor (CPU)
    Returns copy with random frequency and time masking applied.
    """
    x   = x.clone()
    H, W = x.shape[1], x.shape[2]
    fill = x.min().item()

    for _ in range(N_FREQ_MASKS):
        f  = random.randint(0, FREQ_MASK_PARAM)
        f0 = random.randint(0, max(0, H - f))
        x[:, f0 : f0 + f, :] = fill

    for _ in range(N_TIME_MASKS):
        t  = random.randint(0, TIME_MASK_PARAM)
        t0 = random.randint(0, max(0, W - t))
        x[:, :, t0 : t0 + t] = fill

    return x


# ── Dataset ────────────────────────────────────────────────────────────────
class Phase2Dataset(torch.utils.data.Dataset):
    """
    Combines:
      • base  : Phase 1b augmented features (mmap float16, any split)
      • speech: LibriSpeech no_drone windows (optional, train split only)
    Online SpecAugment applied when augment=True.
    """
    def __init__(self, X_base, y_base, X_speech=None, augment=False):
        self.X_base   = X_base                                    # mmap [N, 64, 101]
        self.y_base   = torch.from_numpy(y_base.astype(np.int64))
        self.X_speech = X_speech                                  # [M, 64, 101] or None
        self.augment  = augment
        self.N_base   = len(y_base)
        self.N_speech = len(X_speech) if X_speech is not None else 0

    def __len__(self):
        return self.N_base + self.N_speech

    def __getitem__(self, i):
        if i < self.N_base:
            x = torch.from_numpy(
                    self.X_base[i].astype(np.float32)).unsqueeze(0)
            y = self.y_base[i]
        else:
            j = i - self.N_base
            x = torch.from_numpy(
                    self.X_speech[j].astype(np.float32)).unsqueeze(0)
            y = torch.tensor(1, dtype=torch.long)   # no_drone

        if self.augment and random.random() < SPEC_AUG_PROB:
            x = spec_augment(x)

        return x, y


# ── Class weights ──────────────────────────────────────────────────────────
def compute_class_weights(y_base, n_speech):
    y_all = np.concatenate([y_base, np.ones(n_speech, dtype=np.int8)])
    n, nc = len(y_all), len(CLASSES)
    weights = []
    for k in range(nc):
        nk = int((y_all == k).sum())
        weights.append(n / (nc * nk) if nk else 1.0)
    print("Class weights:")
    for k, w in enumerate(weights):
        nk = int((y_all == k).sum())
        print(f"  {CLASSES[k]:12s}  {nk:>9,} windows  weight={w:.4f}")
    return torch.tensor(weights, dtype=torch.float32)


# ── Training helpers ───────────────────────────────────────────────────────
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


# ── Overall evaluation ─────────────────────────────────────────────────────
def evaluate(model, loader, title="Test"):
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

    print(f"\n=== {title} ===")
    print(f"Accuracy: {acc:.4f}  ({100*acc:.2f}%)")
    print(f"{'Class':12s}  Precision  Recall    FPR       FNR")
    print("-" * 56)
    metrics = {"accuracy": float(acc), "confusion_matrix": C.tolist(),
               "classes": CLASSES}
    for k, cls in enumerate(CLASSES):
        TP = C[k,k]; FP = C[:,k].sum()-TP; FN = C[k,:].sum()-TP; TN = C.sum()-TP-FP-FN
        pr  = TP/(TP+FP) if TP+FP else 0
        re  = TP/(TP+FN) if TP+FN else 0
        fpr = FP/(FP+TN) if FP+TN else 0
        fnr = FN/(FN+TP) if FN+TP else 0
        print(f"{cls:12s}  {pr:.4f}     {re:.4f}    {fpr:.4f}    {fnr:.4f}")
        metrics[cls] = {"precision": pr, "recall": re, "FPR": fpr, "FNR": fnr}
    return metrics, preds, labels


# ── Speech false alarm rate ────────────────────────────────────────────────
def evaluate_speech_far(model, X_speech_test):
    """
    Run the model on held-out speech-only windows.
    Any prediction of DRONE on speech is a false alarm.
    """
    model.eval()
    n_total = len(X_speech_test)
    n_fa    = 0
    BATCH   = 512
    with torch.no_grad():
        for i in range(0, n_total, BATCH):
            batch = torch.from_numpy(
                X_speech_test[i : i + BATCH].astype(np.float32)
            ).unsqueeze(1).to(DEVICE)
            preds = model(batch).argmax(1).cpu().numpy()
            n_fa += int((preds == DRONE_IDX).sum())

    far = n_fa / n_total if n_total else 0.0
    print(f"\n=== Speech False Alarm Rate ===")
    print(f"  Speech windows : {n_total:,}")
    print(f"  False alarms   : {n_fa:,}")
    print(f"  FAR            : {far*100:.2f}%")
    return {"far": float(far), "n_windows": n_total, "n_false_alarms": n_fa}


# ── SNR robustness curve ───────────────────────────────────────────────────
def evaluate_snr_curve(model, X_drone_test):
    """
    Add Gaussian noise at various SNR levels to the drone test features
    (feature-space approximation) and measure recall.
    """
    model.eval()
    print("\n=== SNR Robustness Curve ===")
    print(f"{'Condition':>10}  Drone Recall")
    print("-" * 28)

    X = torch.from_numpy(X_drone_test.astype(np.float32)).unsqueeze(1)  # [N,1,64,101]
    results = {}

    # Clean baseline
    with torch.no_grad():
        p_clean = model(X.to(DEVICE)).argmax(1).cpu().numpy()
    r_clean = (p_clean == DRONE_IDX).mean()
    print(f"{'clean':>10}  {r_clean:.4f}")
    results["clean"] = float(r_clean)

    sig_power = X.pow(2).mean()
    for snr_db in SNR_LEVELS_DB:
        noise_power = sig_power / (10 ** (snr_db / 10.0))
        noise       = torch.randn_like(X) * noise_power.sqrt()
        X_noisy     = X + noise
        with torch.no_grad():
            preds = model(X_noisy.to(DEVICE)).argmax(1).cpu().numpy()
        r = (preds == DRONE_IDX).mean()
        print(f"{snr_db:>+10d}dB  {r:.4f}")
        results[str(snr_db)] = float(r)

    return results


# ── Plot SNR curve ─────────────────────────────────────────────────────────
def plot_snr_curve(snr_results, save_path):
    snrs    = sorted(int(k) for k in snr_results if k != "clean")
    recalls = [snr_results[str(s)] * 100 for s in snrs]
    clean   = snr_results["clean"] * 100

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.axhline(clean, ls="--", color="#1db954",
               label=f"Clean ({clean:.1f}%)")
    ax.plot(snrs, recalls, "o-", color="royalblue", lw=2, ms=6)
    ax.set_xlabel("SNR (dB)  [Gaussian noise on test features]")
    ax.set_ylabel("Drone Recall (%)")
    ax.set_title("Phase 2 v2 — SNR Robustness")
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(str(save_path), dpi=150)
    plt.close(fig)
    print(f"SNR curve saved: {save_path.name}")


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(" Phase 2 v2 -- Speech + Noise Robust Drone Detection")
    print("=" * 60)
    print(f"Device : {DEVICE}"
          + (f"  ({torch.cuda.get_device_name(0)})" if DEVICE.type == "cuda" else ""))

    # ── [1/5] Speech dataset ──────────────────────────────────────
    print("\n[1/5] Preparing speech dataset (LibriSpeech test-clean)...")
    download_librispeech()
    X_sp_train, y_sp_train, X_sp_test = extract_speech_features()
    n_speech_train = len(X_sp_train) if X_sp_train is not None else 0
    print(f"  Speech train windows : {n_speech_train:,}")

    # ── [2/5] Base training features ──────────────────────────────
    print("\n[2/5] Loading base training features (Phase 1b aug or Phase 1)...")
    aug_npy = FEATURES_DIR / "X_train_aug.npy"
    if aug_npy.exists():
        X_base = np.load(str(aug_npy), mmap_mode="r")
        y_base = np.load(str(FEATURES_DIR / "y_train_aug.npy"))
        print(f"  Phase 1b augmented  : {X_base.shape}")
    else:
        X_base = np.load(str(FEATURES_DIR / "X_train.npy"), mmap_mode="r")
        y_base = np.load(str(FEATURES_DIR / "y_train.npy"))
        print(f"  Phase 1b cache missing — using Phase 1 : {X_base.shape}")

    X_val  = np.load(str(FEATURES_DIR / "X_val.npy"))
    y_val  = np.load(str(FEATURES_DIR / "y_val.npy"))
    X_test = np.load(str(FEATURES_DIR / "X_test.npy"))
    y_test = np.load(str(FEATURES_DIR / "y_test.npy"))
    print(f"  Val : {len(X_val):,}   Test : {len(X_test):,}")

    # ── [3/5] DataLoaders ─────────────────────────────────────────
    print("\n[3/5] Building DataLoaders...")
    train_ds = Phase2Dataset(X_base, y_base, X_sp_train, augment=True)
    val_ds   = Phase2Dataset(X_val,  y_val,  augment=False)
    test_ds  = Phase2Dataset(X_test, y_test, augment=False)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=True)

    print(f"  Train batches : {len(train_loader):,}  "
          f"(total {len(train_ds):,} windows)")

    # ── [4/5] Load checkpoint + fine-tune ─────────────────────────
    print("\n[4/5] Fine-tuning from checkpoint...")
    ckpt_path = (MODELS_DIR / "drone_cnn_phase1b.pth"
                 if (MODELS_DIR / "drone_cnn_phase1b.pth").exists()
                 else MODELS_DIR / "drone_cnn_phase1.pth")
    if not ckpt_path.exists():
        raise FileNotFoundError(
            "No base model found. Run train_phase1_gpu.py first.")
    print(f"  Loading : {ckpt_path.name}")

    ckpt  = torch.load(str(ckpt_path), map_location=DEVICE, weights_only=False)
    model = DroneCNN().to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])

    cw        = compute_class_weights(y_base, n_speech_train).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=cw)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS)

    best_val_loss, best_state = float("inf"), None
    t0 = time.time()

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, True)
        va_loss, va_acc = run_epoch(model, val_loader,   criterion, optimizer, False)
        scheduler.step()
        marker = ""
        if va_loss < best_val_loss:
            best_val_loss = va_loss
            best_state = {k: v.cpu().clone()
                          for k, v in model.state_dict().items()}
            marker = "  <- best"
        print(f"  Epoch {epoch:02d}/{EPOCHS}  "
              f"train {tr_loss:.4f}/{tr_acc:.4f}  |  "
              f"val {va_loss:.4f}/{va_acc:.4f}"
              f"  lr={scheduler.get_last_lr()[0]:.2e}{marker}", flush=True)

    print(f"\nFine-tuning done in {(time.time()-t0)/60:.1f} min")

    # ── [5/5] Evaluate ────────────────────────────────────────────
    print("\n[5/5] Evaluating...")
    model.load_state_dict(best_state)

    metrics, preds, labels = evaluate(model, test_loader,
                                       "Phase 2 v2 — Overall Test")

    if X_sp_test is not None:
        far_results = evaluate_speech_far(model, X_sp_test)
        metrics["speech_far"] = far_results

    # SNR curve on drone test windows only
    drone_mask  = (y_test == DRONE_IDX)
    snr_results = evaluate_snr_curve(model, X_test[drone_mask])
    metrics["snr_curve"] = snr_results

    # ── Save model ────────────────────────────────────────────────
    model_path = MODELS_DIR / "drone_cnn_phase2v2.pth"
    torch.save({
        "model_state_dict": best_state,
        "classes"         : CLASSES,
        "config": {
            "n_mels"      : N_MELS,
            "n_fft"       : N_FFT,
            "win_len"     : WIN_LEN,
            "hop_len"     : HOP_LEN,
            "fs"          : FS,
            "win_samples" : WIN_SAMPLES,
            "phase"       : "2v2",
            "speech_aug"  : True,
            "spec_augment": True,
        }
    }, str(model_path))
    print(f"\nModel saved : {model_path}")

    # ── Save results ──────────────────────────────────────────────
    with open(RESULTS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Confusion chart
    fig, ax = plt.subplots(figsize=(6, 5))
    cm_norm = confusion_matrix(labels, preds, normalize="true")
    ConfusionMatrixDisplay(cm_norm, display_labels=CLASSES).plot(
        ax=ax, colorbar=True, cmap="Blues", values_format=".2f")
    ax.set_title("Phase 2 v2 — Test Set")
    fig.tight_layout()
    fig.savefig(str(RESULTS_DIR / "confusion_chart.png"), dpi=150)
    plt.close(fig)
    print(f"Confusion chart saved")

    # SNR curve
    plot_snr_curve(snr_results, RESULTS_DIR / "snr_curve.png")

    # ── Summary ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f" Done!  Test accuracy : {metrics['accuracy']*100:.2f}%")
    if X_sp_test is not None:
        far_pct = metrics["speech_far"]["far"] * 100
        print(f" Speech FAR          : {far_pct:.2f}%  "
              f"(lower is better — was untested in Phase 1b)")
    r_clean  = snr_results["clean"]
    r_neg15  = snr_results.get("-15", 0)
    print(f" SNR robustness      : {r_clean*100:.1f}% clean  ->  "
          f"{r_neg15*100:.1f}% at -15 dB")
    print(f" Model               : {model_path.name}")
    print("=" * 60)
    print("\nTo activate in live_detector.py / predict.py the model is")
    print("auto-selected (phase2v2 > phase1b > phase1) — just restart the app.")


if __name__ == "__main__":
    main()
