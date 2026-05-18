# Codex Task: Phase 2v4 — 5-Specialist CNN Ensemble for Drone Detection

---

## Project Context

This is a drone audio detection project (DADS dataset).
Working directory: `E:\drone_detect`

### Training History (DO NOT MODIFY any of these — add a NEW iteration only)

| Version | File | Notes |
|---------|------|-------|
| phase1 | `models/drone_cnn_phase1.pth` | Baseline |
| phase1b | `models/drone_cnn_phase1b.pth` | Fine-tuned |
| phase2v2 | `models/drone_cnn_phase2v2.pth` | Multi-view |
| phase3 | `models/drone_cnn_phase3.pth` | Phase 3 |
| **phase2v3** | `models/drone_cnn_phase2_v3_multiview_hardnegatives.pth` | **CURRENT BEST** — 92.12% test acc, 0% tank FA, 2.6% engine FA, 0% crowd FA |
| phase2v3b | `models/drone_cnn_phase2_v3b_engine_v2.pth` | v3 retrained with synth_engine_v2 — worse drone recall, not used |

**Rule: Do NOT modify or overwrite any existing file. Create new files only.**

---

## What Phase 2v3 Does (Current System)

Phase 2v3 uses **ONE CNN trained on ALL 5 spectral views simultaneously** (filter augmentation).

```
audio window
    ↓
create_audio_views() → 5 filtered versions
    ↓
During training: pick 1 random view per sample → the single CNN learns all 5
    ↓
During inference: run all 5 views → weighted average → decision
```

The 5 views are:
| # | Name | Filter |
|---|------|--------|
| 0 | raw | none (full band) |
| 1 | HPF-150 | Butterworth 4th-order highpass 150 Hz |
| 2 | HPF-250 | Butterworth 4th-order highpass 250 Hz |
| 3 | BPF-200-6k | Butterworth 4th-order bandpass 200–6000 Hz |
| 4 | BPF-500-6k | Butterworth 4th-order bandpass 500–6000 Hz |

Inference weights: `VIEW_WEIGHTS = [0.05, 0.20, 0.25, 0.35, 0.15]`

---

## What Phase 2v4 Should Do (Your Task)

Train **5 SEPARATE specialist CNNs**, one per view. Each specialist sees ONLY its own filtered version during training. At inference, combine their 5 outputs.

```
audio window
    ↓
create_audio_views() → 5 filtered versions
    ↓                 ↓                  ↓                   ↓                    ↓
raw_model        hpf150_model       hpf250_model       bpf200_model         bpf500_model
  ↓                  ↓                  ↓                   ↓                    ↓
prob_0            prob_1             prob_2              prob_3               prob_4
    ↓
VIEW_WEIGHTS @ [prob_0, prob_1, prob_2, prob_3, prob_4]  → weighted score
    ↓
same decision rule as v3: filteredMax > 0.75  OR  weightedScore > 0.60  OR  voteCount >= 2
```

The key difference from v3:
- v3: one model, filter augmentation (random view per sample)
- v4: five models, each specialist only ever sees its own view

---

## Files to Create

### 1. `E:\drone_detect\train_phase2_v4_specialist.py`

New training script. Do NOT copy-paste and rename v3 — write it clean, but reuse helper functions.

#### Shared constants (copy exactly from v3, do not change values)

```python
FS          = 16000
WIN_SAMPLES = 16000
HOP_SAMPLES = 8000
NOISE_FLOOR = 0.002

_HP150 = sig.butter(4, 150,         btype='high', fs=FS, output='sos')
_HP250 = sig.butter(4, 250,         btype='high', fs=FS, output='sos')
_BP200 = sig.butter(4, [200, 6000], btype='band', fs=FS, output='sos')
_BP500 = sig.butter(4, [500, 6000], btype='band', fs=FS, output='sos')

VIEW_NAMES   = ['raw', 'HPF-150', 'HPF-250', 'BPF-200-6k', 'BPF-500-6k']
VIEW_WEIGHTS = np.array([0.05, 0.20, 0.25, 0.35, 0.15], dtype=np.float32)
```

#### Paths

```python
ROOT        = Path(__file__).parent
MODELS_DIR  = ROOT / "models"
DATA_DIR    = ROOT / "data"
DRONE_DIR   = DATA_DIR / "raw" / "drone"
NODRONE_DIR = DATA_DIR / "raw" / "no_drone"
NOISE_BASE  = DATA_DIR / "noise"
RESULTS_DIR = ROOT / "results" / "phase2_v4"
CKPT_DIR    = RESULTS_DIR / "checkpoints"

# One bundle file containing all 5 specialists
SAVE_PATH = MODELS_DIR / "drone_cnn_phase2_v4_specialist_ensemble.pth"
```

#### Audio helpers (copy exactly from train_phase2_v3.py)

Copy these functions verbatim — do not modify:
- `_norm_view(x)` — peak-normalise a single array
- `create_audio_views(x)` — returns list of 5 filtered views
- `mix_at_snr(clean, noise, snr_db)` — SNR mixing
- `audio_to_logmel(wav)` — returns [64, T] log-mel tensor
- `load_wav(path)` — read WAV mono float32 @ 16 kHz
- `window_audio(audio, win, hop)` — slice into 1-second windows
- `_lp(s, taps)`, `_norm(s, lv)` — convolution LP and normalise
- `synth_tank(n, t0)` — synthetic tank noise
- `synth_engine(n, t0)` — **use v2 version (see below)**
- `synth_crowd(n, t0)` — synthetic crowd noise
- `collect_windows_from_files(folder, max_wins)`
- `collect_noise_windows(noise_type, max_wins)`
- `split_files(files, frac_train, frac_val)`

#### synth_engine v2 (use this exact version)

```python
def synth_engine(n, t0=0.0):
    rng = np.random.default_rng(int(t0 * 1000 + 17) % 99991)
    f0  = rng.uniform(60.0, 120.0)
    t   = np.linspace(t0, t0 + n / FS, n, endpoint=False)
    rpm = 1.0 + 0.05 * np.sin(2 * np.pi * 1.2 * t)
    ph  = np.cumsum(rpm) * (f0 / FS) * 2 * np.pi
    harm = (0.55*np.sin(ph) + 0.25*np.sin(2*ph) + 0.12*np.sin(3*ph) +
            0.06*np.sin(4*ph) + 0.03*np.sin(5*ph))
    exhaust = _lp(rng.standard_normal(n), max(1, int(FS/2000))) * 0.7
    mech = np.zeros(n)
    pos = 0
    while pos < n:
        pos += int(rng.integers(max(1, int(FS*0.03)), max(2, int(FS*0.12))))
        if pos >= n: break
        b = min(int(rng.integers(1, 6)), n - pos)
        if b > 0:
            mech[pos:pos+b] = rng.standard_normal(b) * rng.uniform(0.05, 0.3)
    return _norm(harm + exhaust + mech)
```

#### DroneCNN architecture (copy exactly — do not change)

```python
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
        return self.fc(self.gap(self.features(x)).view(x.size(0), -1))
```

#### Dataset for a single specialist

Unlike v3 (which picks a random view per sample), each specialist dataset uses ONE fixed view index.

```python
class SpecialistDataset(Dataset):
    """
    Each sample uses ONLY the view at `view_idx` (0–4).
    No filter augmentation — this model specialises in exactly one projection.

    Label convention:
      0 = drone   (drone alone, drone+tank, drone+engine, drone+crowd, drone+speech)
      1 = no_drone (tank alone, engine alone, crowd alone, speech alone, pure noise)
    """
    def __init__(self, view_idx, drone_wins, noise_wins, nodrone_wins,
                 n_drone, n_nodrone, snr_levels, augment=True):
        self.view_idx     = view_idx
        self.drone_wins   = drone_wins
        self.noise_wins   = noise_wins
        self.nodrone_wins = nodrone_wins
        self.n_drone      = n_drone
        self.n_nodrone    = n_nodrone
        self.snr_levels   = snr_levels
        self.augment      = augment
        self.noise_types  = [k for k, v in noise_wins.items() if v]

    def __len__(self):
        return self.n_drone + self.n_nodrone

    def __getitem__(self, idx):
        if idx < self.n_drone:
            # DRONE — mix with random noise 80% of the time
            dw    = self.drone_wins[idx % len(self.drone_wins)].copy()
            audio = dw
            if self.augment and self.noise_types:
                if random.random() < 0.80:
                    ntype = random.choice(self.noise_types)
                    nwin  = self.noise_wins[ntype][
                                random.randrange(len(self.noise_wins[ntype]))]
                    snr   = random.choice(self.snr_levels)
                    audio = mix_at_snr(dw, nwin, snr)
                    # 20% chance: second noise layer
                    if random.random() < 0.20 and len(self.noise_types) > 1:
                        ntype2 = random.choice(
                            [t for t in self.noise_types if t != ntype])
                        nwin2  = self.noise_wins[ntype2][
                                     random.randrange(len(self.noise_wins[ntype2]))]
                        audio  = mix_at_snr(audio, nwin2, random.choice(self.snr_levels))
            label = 0  # drone
        else:
            # NO-DRONE — pure noise (hard negative)
            nd_idx = (idx - self.n_drone) % len(self.nodrone_wins)
            audio  = self.nodrone_wins[nd_idx].copy()
            label  = 1  # no_drone

        # Apply ONLY this specialist's view (no random selection)
        views      = create_audio_views(audio)
        audio_view = views[self.view_idx]

        logmel = audio_to_logmel(audio_view)   # [64, T]
        return logmel.unsqueeze(0).float(), torch.tensor(label, dtype=torch.long)
```

#### Training loop

- Train 5 models sequentially (one per view_idx = 0..4)
- Each model: same hyperparameters as v3 (epochs=50, batch=32, lr=1e-3)
- Early stopping: patience=8 epochs on val_acc (same as v3)
- ReduceLROnPlateau on val_loss, factor=0.5, patience=4, min_lr=1e-5
  - **Important**: Do NOT pass `verbose=True` to ReduceLROnPlateau — newer PyTorch removed it
- Save the best checkpoint per specialist (by val_acc) in `results/phase2_v4/checkpoints/best_{view_name}.pth`
- After training all 5, bundle them:

```python
bundle = {
    'phase':         'phase2v4',
    'hpf_hz':        0,
    'mel_fmin':      0.0,
    'view_names':    VIEW_NAMES,
    'view_weights':  VIEW_WEIGHTS.tolist(),
    'drone_idx':     0,   # class index for "drone"
}
for vi, vname in enumerate(VIEW_NAMES):
    key = f'model_{vi}_{vname.replace("-","_").replace("+","_")}'
    bundle[key] = best_state_dicts[vi]   # OrderedDict from model.state_dict()
torch.save(bundle, SAVE_PATH)
print(f"Saved ensemble bundle -> {SAVE_PATH}")
```

#### Post-training condition tests

After training, run the same 8 condition scenarios as `internal_test.py`:
1. Pure drone (from DADS test set)
2. Drone + tank
3. Drone + engine
4. Drone + crowd
5. Tank alone → expect 0% false alarm
6. Engine alone → expect 0% false alarm
7. Crowd alone → expect 0% false alarm
8. Pure noise → expect 0% false alarm

For each scenario, run N_CHUNKS=600 windows (600 seconds synthetic).
For each window, pass it through all 5 specialists:

```python
@torch.no_grad()
def predict_ensemble(models, audio, drone_idx, device):
    """Run 5 specialist models on their respective views. Return weighted score."""
    views = create_audio_views(audio)
    probs = np.zeros(5, dtype=np.float32)
    for vi, (model, view) in enumerate(zip(models, views)):
        lm = audio_to_logmel(view)
        X  = lm.unsqueeze(0).unsqueeze(0).to(device)  # [1,1,64,T]
        sc = torch.softmax(model(X), dim=1)
        probs[vi] = sc[0, drone_idx].item()
    return float((VIEW_WEIGHTS * probs).sum()), probs

FMAX_THR  = 0.75
SCORE_THR = 0.60
VOTE_THR  = 0.60
VOTES_NEED = 2

def is_detection(probs):
    ws  = float(VIEW_WEIGHTS @ probs)
    fm  = float(probs[1:].max())        # filtered views only (exclude raw)
    vc  = int((probs > VOTE_THR).sum())
    return (fm > FMAX_THR) or (ws > SCORE_THR) or (vc >= VOTES_NEED), ws, fm, vc
```

Print results in same format as internal_test.py — detection rate %, mean weighted score, PASS/WARN/FAIL verdict.

#### Command-line arguments

```
python train_phase2_v4_specialist.py              # full training (50 epochs × 5 models)
python train_phase2_v4_specialist.py --quick      # 500 examples/class, 5 epochs (pipeline test)
python train_phase2_v4_specialist.py --epochs 30
python train_phase2_v4_specialist.py --no-gpu
```

---

### 2. `E:\drone_detect\compare_v3_v4.py`

Side-by-side comparison script. Loads both models and runs the same 8 scenarios on each.

```python
# Loads:
#   v3:  models/drone_cnn_phase2_v3_multiview_hardnegatives.pth  (single model, multiview)
#   v4:  models/drone_cnn_phase2_v4_specialist_ensemble.pth      (5-model bundle)
# Prints a table like:
#
# Scenario               | v3 Det% | v3 ws  | v4 Det% | v4 ws  | Winner
# -----------------------+---------+--------+---------+--------+-------
# drone alone            |  97.3%  | 0.832  |  xx.x%  | x.xxx  |  ???
# drone+tank             |  98.1%  | 0.871  |  ...    |        |
# ...
```

Reuse the same synth functions from v3 (copy or import them).

---

### 3. Update `E:\drone_detect\live_detector.py`

**Do NOT remove anything**. Add v4 ensemble support alongside the existing v3 path.

#### Additions needed

1. Update `_best_model()` to prefer v4 ensemble over v3:
   ```python
   PRIORITY = [
       'drone_cnn_phase2_v4_specialist_ensemble',   # NEW — try first
       'drone_cnn_phase2_v3_multiview_hardneg',
       'drone_cnn_phase2_v3b_engine_v2',
       'drone_cnn_phase3',
       'drone_cnn_phase2v2',
   ]
   ```
   (match by substring of filename)

2. Add a `USE_ENSEMBLE` flag (analogous to existing `USE_MULTIVIEW`):
   ```python
   USE_ENSEMBLE = 'v4_specialist' in best_model_path.stem
   ```

3. Add `load_ensemble(path, device)` function that:
   - loads the bundle .pth
   - reconstructs 5 DroneCNN instances from the 5 state dicts
   - returns `(models_list, drone_idx, view_weights)`

4. Add `infer_ensemble(models, drone_idx, window)` function that:
   - checks `np.abs(window).max() < NOISE_FLOOR` → return silent
   - calls `predict_ensemble()` (same logic as condition tests above)
   - applies same decision rule (FMAX_THR/SCORE_THR/VOTE_THR)
   - returns `(weighted_score, detected_bool, mode_string)`

5. In `InferenceThread` and `TestInferenceThread`:
   - if `USE_ENSEMBLE`: call `infer_ensemble()`
   - elif `USE_MULTIVIEW`: call existing `infer_multiview()`
   - else: call existing single-model path

6. Update the info bar label to show "specialist-5" when running v4.

**Keep all existing v3 / multiview code unchanged.**

---

## Label Convention Summary

```
drone alone            → label 0  (drone)
drone + tank           → label 0  (drone)
drone + engine         → label 0  (drone)
drone + crowd          → label 0  (drone)
drone + speech         → label 0  (drone)

tank alone             → label 1  (no_drone)
engine alone           → label 1  (no_drone)
crowd alone            → label 1  (no_drone)
speech alone           → label 1  (no_drone)
pure noise / silence   → label 1  (no_drone)
```

---

## Data Paths

```
E:\drone_detect\data\raw\drone\          ← ~163,591 WAV files (DADS dataset, 16 kHz)
E:\drone_detect\data\raw\no_drone\       ← ~16,729 WAV files
E:\drone_detect\data\noise\tank\         ← may be empty (uses synth_tank fallback)
E:\drone_detect\data\noise\engine\       ← may be empty (uses synth_engine v2 fallback)
E:\drone_detect\data\noise\crowd\        ← may be empty (uses synth_crowd fallback)
E:\drone_detect\data\noise\wind\         ← may be empty
E:\drone_detect\data\noise\speech\       ← may be empty
```

---

## Expected Outcome

- 5 specialist models trained and saved in one bundle
- Condition test results printed per specialist AND ensemble
- compare_v3_v4.py runs cleanly and prints comparison table
- live_detector.py works with v4 ensemble when file is present, falls back to v3 if not
- Nothing deleted or overwritten

---

## Quick Sanity Check

After writing the files, run:
```
cd E:\drone_detect
python train_phase2_v4_specialist.py --quick
```
Expected: all 5 models train in ~5 min, saves bundle, prints condition test table.
If `--quick` passes cleanly, the full training (`--epochs 50`) can be run separately.
